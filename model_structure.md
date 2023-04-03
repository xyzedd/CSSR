The implemented CSSR model is structured in the following way:
The main important classes are implemented in cssr.py are:
- Backbone:
    This class consists of the backbone network which is wideresnet structure, a wider version of original resnet. The network structure is changed to BN-ReLU-Conv from Conv-BN-ReLU for faster training. In the wideresnet, there are BasicBlock and NetworkBlock where the width of the network block is nb_layer parameter. In the Backbone class, the type of wideresnet is chosen from the config file.

- Autoencoder:
    This is an autoencoder class which is applied to the feature map obtained from Backbone network. The input channel, hidden layers, and latente dimension all are provided from the config files.
- CSSRClassifier:
    This CSSRClassifier class is used for the classification of the input where the logits are the reconstruction error obtained from input and its corresponding manifold of the autoencoder. The error is calculated using the L1 norm (also called Manhattan distance) and these error values are treated as logits before feeding into the SoftMax classifier.
- BackboneandClassifier:
    This class incorporates both Backbone network and the CSSRClassifier network and merges into a single pipeline. There also exists other classifier called Linear classifier and is chosen according to the config file.
- CSSRModel:
    This class represents the whole end to end model ready for training. The logits obtained from CSSR classifier are then classified to the known class samples using Softmax.