"""

                _modelpicker.py


GUI code for choosing a model

"""
#import param
import panel as pn
import tensorflow as tf
from patchwork._fine_tuning_models import GlobalPooling, ConvNet, MultiscalePooling
from patchwork._output_models import SigmoidCrossEntropy, CosineOutput, SigmoidFocalLoss
from patchwork.feature._moco import copy_model

class ModelPicker(object):
    """
    For building fine-tuning models
    """
    def __init__(self, num_classes, feature_shape, pw):#, feature_extractor=None):
        """

        """
        fine_tuning_model_dict = {"Global Pooling":GlobalPooling(), "Convnet":ConvNet(),
                                  "Multiscale Pooling":MultiscalePooling()}
        output_model_dict = {"Sigmoid Cross-entropy":SigmoidCrossEntropy(),
                    "Sigmoid Focal Loss":SigmoidFocalLoss()}

        self._current_model = pn.pane.Markdown("**No current model**\n")
        self._feature_shape = feature_shape
        self._num_classes = num_classes
        #self._inpt_channels = inpt_channels
        self._pw = pw
        #self._feature_extractor = feature_extractor
        self._weight_decay = pn.widgets.LiteralInput(name="Weight decay", value=0., type=float)
        # ------- FINE TUNING MODEL SELECTION AND CONFIGURATION -------
        # dropdown and watcher to pick model type
        self._fine_tuning_chooser = pn.widgets.Select(options=fine_tuning_model_dict)
        self._fine_tuning_watcher = self._fine_tuning_chooser.param.watch(self._fine_tuning_callback,
                                                                 ["value"])
        # model description pane
        self._fine_tuning_desc = pn.pane.Markdown(self._fine_tuning_chooser.value.description)
        # model hyperparameters
        self._fine_tuning_hyperparams = pn.panel(self._fine_tuning_chooser.value)

        # ------- OUTPUT MODEL SELECTION AND CONFIGURATION -------
        # dropdown and watcher to pick model type
        self._output_chooser = pn.widgets.Select(options=output_model_dict)
        self._output_watcher = self._output_chooser.param.watch(self._output_callback,
                                                                 ["value"])
        # model description pane
        self._output_desc = pn.pane.Markdown(self._output_chooser.value.description)
        # model hyperparameters
        self._output_hyperparams = pn.panel(self._output_chooser.value)

        # model-builder button
        self._build_button = pn.widgets.Button(name="Build Model")
        self._build_button.on_click(self._build_callback)

        # --------- SEMI-SUPERVISED LEARNING INTERFACE -------
        semisup1 = "### Semi-supervised learning\n\n"
        semisup2 = "Use unlabeled samples to guide decision boundary"
        semisup_widgets = {
            "header":pn.pane.Markdown(semisup1+semisup2),
            "choosetype":pn.widgets.Select(options=["FixMatch", "Domain Confusion"]),
            "lambda":pn.widgets.LiteralInput(name='loss weight lambda (0 to disable)', value=0., type=float),
            "tau":pn.widgets.LiteralInput(name="threshold tau (FixMatch only)", value=0.95, type=float),
            "mu":pn.widgets.LiteralInput(name="batch size multiplier mu", value=1, type=int)
            }
        self._semisup = semisup_widgets
        self._semisup_panel = pn.Column(*[semisup_widgets[x] for x in
                                          ["header", "choosetype", "lambda", "tau",
                                           "mu"]])


    def panel(self):
        """
        Build the GUI as a panel object
        """
        fine_tuning = pn.Column(
            pn.pane.Markdown("### Fine-tuning model\nMap feature tensor to a dense vector"),
            self._fine_tuning_chooser,
            self._fine_tuning_desc,
            self._fine_tuning_hyperparams,
            self._weight_decay
        )
        output = pn.Column(
            pn.pane.Markdown("### Output model\nMap dense vector to class probabilities"),
            self._output_chooser,
            self._output_desc,
            self._output_hyperparams
        )
        """
        semisupervised = pn.Column(
            pn.pane.Markdown("### Semi-supervised learning\nUse unlabeled images to guide decision boundaries."),
            self._entropy_reg,
            #self._mean_teacher_alpha
        )"""

        return pn.Column(
            pn.pane.Markdown("## Model Options"),
            pn.Row(fine_tuning,
                   pn.Spacer(background="whitesmoke", width=10),
                   output,
                   pn.Spacer(background="whitesmoke", width=10),
                   self._semisup_panel),
            pn.Row(self._build_button, self._current_model)
        )

    def _fine_tuning_callback(self, *events):
        # update description and hyperparameters
        self._fine_tuning_desc.object = self._fine_tuning_chooser.value.description
        if hasattr(pn.panel(self._fine_tuning_chooser.value), "objects"):
            self._fine_tuning_hyperparams.objects = pn.panel(self._fine_tuning_chooser.value).objects
        else:

            self._fine_tuning_hyperparams.objects = []

    def _output_callback(self, *events):
        # update description and hyperparameters
        self._output_desc.object = self._output_chooser.value.description
        if hasattr(pn.panel(self._output_chooser.value), "objects"):
            self._output_hyperparams.objects = pn.panel(self._output_chooser.value).objects
        else:

            self._output_hyperparams.objects = []

    def _build_callback(self, *events):
        """
        When you hit the build button:
            0) make a backup copy of the feature extractor
            1) build the fine-tuning model
            2) build the output model and loss function
            3) generate the full end-to-end model for inference
            4) if doing mean-teacher semi-supervision, set up teacher model
            5) reset the lists for recording training loss in the GUI object
        """
        with self._pw._strategy.scope():
            # 0) copy the feature extractor
            if self._pw.models["feature_extractor_backup"] is not None:
                #self._pw.models["feature_extractor"] = copy_model(self._pw.models["feature_extractor_backup"])
                feature_extractor = copy_model(self._pw.models["feature_extractor_backup"])
            else:
                feature_extractor = None
            # 1) BUILD THE FINE-TUNING MODEL (this may build a modified feature extractor)
            fine_tuning_model, feature_extractor = self._fine_tuning_chooser.value.build(self._feature_shape,
                                                                                         feature_extractor)
            self._pw.models["feature_extractor"] = feature_extractor
            tuning_output_channels = fine_tuning_model.output_shape[-1]
            # 2) BUILD THE OUTPUT MODEL AND LOSS FUNCTION
            output_model, loss = self._output_chooser.value.build(self._num_classes,
                                                                      tuning_output_channels)
            self._pw.loss_fn = loss

        curr_finetune = "**Current fine-tuning model:** %s (%s parameters)"%(self._fine_tuning_chooser.value.name,
                                                                            fine_tuning_model.count_params())
        curr_output = "**Current output model:** %s (%s parameters)"%(self._output_chooser.value.name,
                                                                            output_model.count_params())
        self._current_model.object = curr_finetune + "\n\n" + curr_output
        # update GUI object's model dictionary
        self._pw.models["fine_tuning"] = fine_tuning_model
        self._pw.models["output"] = output_model
        # record hyperparameter info
        self._pw._model_params["fine_tuning"] = self._fine_tuning_chooser.value.model_params()
        self._pw._model_params["output"] = self._output_chooser.value.model_params()

        if self._semisup["choosetype"].value == "FixMatch":
            fm_lambda = self._semisup["lambda"].value
            domain_lambda = 0
        else:
            domain_lambda = self._semisup["lambda"].value
            fm_lambda = 0
        self._pw._model_params["semisup"] = {
                "fixmatch_weight_lambda":fm_lambda,
                "domain_confusion_weight_lambda":domain_lambda,
                "fixmatch_tau":self._semisup["tau"].value,
                "batch_size_multiplier_mu":self._semisup["mu"].value}
        self._pw._model_params["weight_decay"] = self._weight_decay.value


        # 3) GENERATE FULL MODEL (for inference)
        with self._pw._strategy.scope():
            #feature_extractor = self._pw.models["feature_extractor"]
            if feature_extractor is not None:
                inpt = tf.keras.layers.Input(feature_extractor.input_shape[1:])
                net = feature_extractor(inpt)
            else:
                inpt = tf.keras.layers.Input(fine_tuning_model.input_shape[1:])
                net = inpt
            net = fine_tuning_model(net)
            net = output_model(net)
            self._pw.models["full"] = tf.keras.Model(inpt, net)

        if self._semisup["choosetype"].value == "FixMatch":
            self._pw._semi_supervised = (self._semisup["lambda"].value > 0)
            self._pw._domain_adapt = False
        else:
            self._pw._domain_adapt = (self._semisup["lambda"].value > 0)
            self._pw._semi_supervised = False

        # 5) RESET LOSS RECORDERS
        self._pw.training_loss = []
        self._pw.semisup_loss = []
        self._pw.test_loss = []
        self._pw.test_loss_step = []




