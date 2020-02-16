# From https://github.com/albermax/innvestigate/blob/master/innvestigate/utils/keras/graph.py
#
# COPYRIGHT
#
# The copyright in this software is being made available under the BSD
# License, included below. This software is subject to other contributor rights,
# including patent rights, and no such rights are granted under this license.
#
# All contributions by TU Berlin:
# Copyright (c) 2017-2018 Maximilian Alber, Miriam Haegele, Philipp Seegerer, Kristof T. Schuett
# All rights reserved.
#
# All contributions by Fraunhofer Heinrich Hertz Institute:
# Copyright (c) 2018, Sebastian Lapuschkin
# All rights reserved. Patent pending.
#
# All other contributions:
# Copyright (c) 2018, the respective contributors.
# All rights reserved.
#
# Each contributor holds copyright over their respective contributions.
#
# LICENSE
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


# Get Python six functionality:
from __future__ import\
    absolute_import, print_function, division, unicode_literals
from builtins import range, zip
import six


###############################################################################
###############################################################################
###############################################################################


import inspect
import keras.backend as K
import keras.engine.topology
import keras.layers
import keras.models
import numpy as np


__all__ = [
    "get_kernel",

    "get_layer_inbound_count",
    "get_layer_outbound_count",
    "copy_layer_wo_activation",
    "copy_layer",
    "pre_softmax_tensors",
    "model_wo_softmax",

    "get_model_layers",
    "model_contains",

    "trace_model_execution",
    "get_model_execution_trace",
    "get_model_execution_graph",
    "print_model_execution_graph",

    "get_bottleneck_nodes",
    "get_bottleneck_tensors",

    "ReverseMappingBase",
    "reverse_model",
]


###############################################################################
###############################################################################
###############################################################################


def to_list(l):
    """ If not list, wraps parameter into a list."""
    if not isinstance(l, list):
        return [l, ]
    else:
        return l


def contains_activation(layer, activation=None):
    """
    Check whether the layer contains an activation function.
    activation is None then we only check if layer can contain an activation.
    """

    # todo: add test and check this more throughroughly.
    # rely on Keras convention.
    if hasattr(layer, "activation"):
        if activation is not None:
            return layer.activation == keras.activations.get(activation)
        else:
            return True
    elif isinstance(layer, keras.layers.ReLU):
        if activation is not None:
            return (keras.activations.get("relu") ==
                    keras.activations.get(activation))
        else:
            return True
    elif isinstance(layer, (
            keras.layers.advanced_activations.ELU,
            keras.layers.advanced_activations.LeakyReLU,
            keras.layers.advanced_activations.PReLU,
            keras.layers.advanced_activations.Softmax,
            keras.layers.advanced_activations.ThresholdedReLU)):
        if activation is not None:
            raise Exception("Cannot detect activation type.")
        else:
            return True
    else:
        return False



def is_network(layer):
    """
    Is network in network?
    """
    return isinstance(layer, keras.engine.topology.Network)

###############################################################################
###############################################################################
###############################################################################


def get_kernel(layer):
    """Returns the kernel weights of a layer, i.e, w/o biases."""
    ret = [x for x in layer.get_weights() if len(x.shape) > 1]
    assert len(ret) == 1
    return ret[0]


def get_input_layers(layer):
    """Returns all layers that created this layer's inputs."""
    ret = set()

    for node_index in range(len(layer._inbound_nodes)):
        Xs = to_list(layer.get_input_at(node_index))
        for X in Xs:
            ret.add(X._keras_history[0])

    return ret


###############################################################################
###############################################################################
###############################################################################


def get_layer_inbound_count(layer):
    """Returns the number inbound nodes of a layer."""
    return len(layer._inbound_nodes)


def get_layer_outbound_count(layer):
    """Returns the number outbound nodes of a layer."""
    return len(layer.outbound_nodes)


def get_symbolic_weight_names(layer, weights=None):
    """Attribute names for weights

    Looks up the attribute names of weight tensors.

    :param layer: Targeted layer.
    :param weights: A list of weight tensors.
    :return: The attribute names of the weights.
    """

    if weights is None:
        weights = layer.weights

    good_guesses = [
        "kernel",
        "bias",
        "gamma",
        "beta",
        "moving_mean",
        "moving_variance",
        "depthwise_kernel",
        "pointwise_kernel"
    ]

    ret = []
    for weight in weights:
        for attr_name in good_guesses+dir(layer):
            if(hasattr(layer, attr_name) and
               id(weight) == id(getattr(layer, attr_name))):
                ret.append(attr_name)
                break
    if len(weights) != len(ret):
        raise Exception("Could not find symoblic weight name(s).")

    return ret


def update_symbolic_weights(layer, weight_mapping):
    """Updates the symbolic tensors of a layer

    Updates the symbolic tensors of a layer by replacing them.

    Note this does not update the loss or anything alike!
    Use with caution!

    :param layer: Targeted layer.
    :param weight_mapping: Dict with attribute name and weight tensors
      as keys and values.
    """

    trainable_weight_ids = [id(x) for x in layer._trainable_weights]
    non_trainable_weight_ids = [id(x) for x in layer._non_trainable_weights]

    for name, weight in six.iteritems(weight_mapping):
        current_weight = getattr(layer, name)
        current_weight_id = id(current_weight)

        if current_weight_id in trainable_weight_ids:
            idx = trainable_weight_ids.index(current_weight_id)
            layer._trainable_weights[idx] = weight
        else:
            idx = non_trainable_weight_ids.index(current_weight_id)
            layer._non_trainable_weights[idx] = weight

        setattr(layer, name, weight)


def get_layer_from_config(old_layer,
                          new_config,
                          weights=None,
                          reuse_symbolic_tensors=True):
    """Creates a new layer from a config

    Creates a new layer given a changed config and weights etc.

    :param old_layer: A layer that shall be used as base.
    :param new_config: The config to create the new layer.
    :param weights: Weights to set in the new layer.
      Options: np tensors, symbolic tensors, or None,
      in which case the weights from old_layers are used.
    :param reuse_symbolic_tensors: If the weights of the
      old_layer are used copy the symbolic ones or copy
      the Numpy weights.
    :return: The new layer instance.
    """
    new_layer = old_layer.__class__.from_config(new_config)

    if weights is None:
        if reuse_symbolic_tensors:
            weights = old_layer.weights
        else:
            weights = old_layer.get_weights()

    if len(weights) > 0:
        input_shapes = old_layer.get_input_shape_at(0)
        # todo: inspect and set initializers to something fast for speedup
        new_layer.build(input_shapes)

        is_np_weight = [isinstance(x, np.ndarray) for x in weights]
        if all(is_np_weight):
            new_layer.set_weights(weights)
        else:
            if any(is_np_weight):
                raise ValueError("Expect either all weights to be "
                                 "np tensors or symbolic tensors.")

            symbolic_names = get_symbolic_weight_names(old_layer)
            update = {name: weight
                      for name, weight in zip(symbolic_names, weights)}
            update_symbolic_weights(new_layer, update)

    return new_layer


def copy_layer_wo_activation(layer,
                             keep_bias=True,
                             name_template=None,
                             weights=None,
                             reuse_symbolic_tensors=True,
                             **kwargs):
    """Copy a Keras layer and remove the activations

    Copies a Keras layer but remove potential activations.

    :param layer: A layer that should be copied.
    :param keep_bias: Keep a potential bias.
    :param weights: Weights to set in the new layer.
      Options: np tensors, symbolic tensors, or None,
      in which case the weights from old_layers are used.
    :param reuse_symbolic_tensors: If the weights of the
      old_layer are used copy the symbolic ones or copy
      the Numpy weights.
    :return: The new layer instance.
    """
    config = layer.get_config()
    if name_template is None:
        config["name"] = None
    else:
        config["name"] = name_template % config["name"]
    if contains_activation(layer):
        config["activation"] = None
    if keep_bias is False and config.get("use_bias", True):
        config["use_bias"] = False
        if weights is None:
            if reuse_symbolic_tensors:
                weights = layer.weights[:-1]
            else:
                weights = layer.get_weights()[:-1]
    return get_layer_from_config(layer, config, weights=weights, **kwargs)


def copy_layer(layer,
               keep_bias=True,
               name_template=None,
               weights=None,
               reuse_symbolic_tensors=True,
               **kwargs):
    """Copy a Keras layer

    Copies a Keras layer.

    :param layer: A layer that should be copied.
    :param keep_bias: Keep a potential bias.
    :param weights: Weights to set in the new layer.
      Options: np tensors, symbolic tensors, or None,
      in which case the weights from old_layers are used.
    :param reuse_symbolic_tensors: If the weights of the
      old_layer are used copy the symbolic ones or copy
      the Numpy weights.
    :return: The new layer instance.
    """
    config = layer.get_config()
    if name_template is None:
        config["name"] = None
    else:
        config["name"] = name_template % config["name"]
    if keep_bias is False and config.get("use_bias", True):
        config["use_bias"] = False
        if weights is None:
            if reuse_symbolic_tensors:
                weights = layer.weights[:-1]
            else:
                weights = layer.get_weights()[:-1]
    return get_layer_from_config(layer, config, weights=weights, **kwargs)


def pre_softmax_tensors(Xs, should_find_softmax=True):
    """Finds the tensors that were preceeding a potential softmax."""
    softmax_found = False

    Xs = to_list(Xs)
    ret = []
    for x in Xs:
        layer, node_index, tensor_index = x._keras_history
        if contains_activation(layer, activation="softmax"):
            softmax_found = True
            if isinstance(layer, keras.layers.Activation):
                ret.append(layer.get_input_at(node_index))
            else:
                layer_wo_act = copy_layer_wo_activation(layer)
                ret.append(layer_wo_act(layer.get_input_at(node_index)))

    if should_find_softmax and not softmax_found:
        raise Exception("No softmax found.")

    return ret


def model_wo_softmax(model):
    """Creates a new model w/o the final softmax activation."""
    return keras.models.Model(inputs=model.inputs,
                              outputs=pre_softmax_tensors(model.outputs),
                              name=model.name)


###############################################################################
###############################################################################
###############################################################################


def get_model_layers(model):
    """Returns all layers of a model."""
    ret = []

    def collect_layers(container):
        for layer in container.layers:
            assert layer not in ret
            ret.append(layer)
            if is_network(layer):
                collect_layers(layer)
    collect_layers(model)

    return ret


def model_contains(model, layer_condition, return_only_counts=False):
    if callable(layer_condition):
        layer_condition = [layer_condition, ]
        single_condition = True
    else:
        single_condition = False

    layers = get_model_layers(model)
    collected_layers = []
    for condition in layer_condition:
        tmp = [layer for layer in layers if condition(layer)]
        collected_layers.append(tmp)
    if return_only_counts is True:
        collected_layers = [len(v) for v in collected_layers]

    if single_condition is True:
        return collected_layers[0]
    else:
        return collected_layers


###############################################################################
###############################################################################
###############################################################################


def trace_model_execution(model, reapply_on_copied_layers=False):
    """
    Trace and linearize excecution of a model and it's possible containers.
    Return a triple with all layers, a list with a linearized execution
    with (layer, input_tensors, output_tensors), and, possible regenerated,
    outputs of the exectution.

    :param model: A kera model.
    :param reapply_on_copied_layers: If the execution needs to be linearized,
      reapply with copied layers. Might be slow. Prevents changes of the
      original layer's node lists.
    """

    # Get all layers in model.
    layers = get_model_layers(model)

    # Check if some layers are containers.
    # Ignoring the outermost container, i.e. the passed model.
    contains_container = any([((l is not model) and is_network(l))
                              for l in layers])

    # If so rebuild the graph, otherwise recycle computations,
    # and create executed node list. (Keep track of paths?)
    if contains_container is True:
        # When containers/models are used as layers, then layers
        # inside the container/model do not keep track of nodes.
        # This makes it impossible to iterate of the nodes list and
        # recover the input output tensors. (see else clause)
        #
        # To recover the computational graph we need to re-apply it.
        # This implies that the tensors-object we use for the forward
        # pass are different to the passed model. This it not the case
        # for the else clause.
        #
        # Note that reapplying the model does only change the inbound
        # and outbound nodes of the model itself. We copy the model
        # so the passed model should not be affected from the
        # reapplication.
        executed_nodes = []

        # Monkeypatch the call function in all the used layer classes.
        monkey_patches = [(layer, getattr(layer, "call")) for layer in layers]
        try:
            def patch(self, method):
                if hasattr(method, "__patched__") is True:
                    raise Exception("Should not happen as we patch "
                                    "objects not classes.")

                def f(*args, **kwargs):
                    input_tensors = args[0]
                    output_tensors = method(*args, **kwargs)
                    executed_nodes.append((self,
                                           input_tensors,
                                           output_tensors))
                    return output_tensors
                f.__patched__ = True
                return f

            # Apply the patches.
            for layer in layers:
                setattr(layer, "call", patch(layer, getattr(layer, "call")))

            # Trigger reapplication of model.
            model_copy = keras.models.Model(inputs=model.inputs,
                                            outputs=model.outputs)
            outputs = to_list(model_copy(model.inputs))
        finally:
            # Revert the monkey patches
            for layer, old_method in monkey_patches:
                setattr(layer, "call", old_method)

        # Now we have the problem that all the tensors
        # do not have a keras_history attribute as they are not part
        # of any node. Apply the flat model to get it.
        from . import apply as kapply
        new_executed_nodes = []
        tensor_mapping = {tmp: tmp for tmp in model.inputs}
        if reapply_on_copied_layers is True:
            layer_mapping = {layer: copy_layer(layer) for layer in layers}
        else:
            layer_mapping = {layer: layer for layer in layers}

        for layer, Xs, Ys in executed_nodes:
            layer = layer_mapping[layer]
            Xs, Ys = to_list(Xs), to_list(Ys)

            if isinstance(layer, keras.layers.InputLayer):
                # Special case. Do nothing.
                new_Xs, new_Ys = Xs, Ys
            else:
                new_Xs = [tensor_mapping[x] for x in Xs]
                new_Ys = to_list(kapply(layer, new_Xs))

            tensor_mapping.update({k: v for k, v in zip(Ys, new_Ys)})
            new_executed_nodes.append((layer, new_Xs, new_Ys))

        layers = [layer_mapping[layer] for layer in layers]
        outputs = [tensor_mapping[x] for x in outputs]
        executed_nodes = new_executed_nodes
    else:
        # Easy and safe way.
        reverse_executed_nodes = [
            (node.outbound_layer, node.input_tensors, node.output_tensors)
            for depth in sorted(model._nodes_by_depth.keys())
            for node in model._nodes_by_depth[depth]
        ]
        outputs = model.outputs

        executed_nodes = reversed(reverse_executed_nodes)

    # This list contains potentially nodes that are not part
    # final execution graph.
    # E.g., a layer was also applied outside of the model. Then its
    # node list contains nodes that do not contribute to the model's output.
    # Those nodes are filtered here.
    used_as_input = [x for x in outputs]
    tmp = []
    for l, Xs, Ys in reversed(list(executed_nodes)):
        if all([y in used_as_input for y in Ys]):
            used_as_input += Xs
            tmp.append((l, Xs, Ys))
    executed_nodes = list(reversed(tmp))

    return layers, executed_nodes, outputs


def get_model_execution_trace(model,
                              keep_input_layers=False,
                              reapply_on_copied_layers=False):
    """
    Returns a list representing the execution graph.
    Each key is the node's id as it is used by the reverse_model method.

    Each associated value contains a dictionary with the following items:

    * nid: the node id.
    * layer: the layer creating this node.
    * Xs: the input tensors (only valid if not in a nested container).
    * Ys: the output tensors (only valid if not in a nested container).
    * Xs_nids: the ids of the nodes creating the Xs.
    * Ys_nids: the ids of nodes using the according output tensor.
    * Xs_layers: the layer that created the accodring input tensor.
    * Ys_layers: the layers using the according output tensor.

    :param model: A kera model.
    :param keep_input_layers: Keep input layers.
    :param reapply_on_copied_layers: If the execution needs to be linearized,
      reapply with copied layers. Might be slow. Prevents changes of the
      original layer's node lists.
    """
    _, execution_trace, _ = trace_model_execution(
        model,
        reapply_on_copied_layers=reapply_on_copied_layers)

    # Enrich trace with node ids.
    current_nid = 0
    tmp = []
    for l, Xs, Ys in execution_trace:
        if isinstance(l, keras.layers.InputLayer):
            tmp.append((None, l, Xs, Ys))
        else:
            tmp.append((current_nid, l, Xs, Ys))
            current_nid += 1
    execution_trace = tmp

    # Create lookups from tensor to creating or receiving layer-node
    inputs_to_node = {}
    outputs_to_node = {}
    for nid, l, Xs, Ys in execution_trace:
        if nid is not None:
            for X in Xs:
                Xid = id(X)
                if Xid in inputs_to_node:
                    inputs_to_node[Xid].append(nid)
                else:
                    inputs_to_node[Xid] = [nid]

        if keep_input_layers or nid is not None:
            for Y in Ys:
                Yid = id(Y)
                if Yid in inputs_to_node:
                    raise Exception("Cannot be more than one creating node.")
                outputs_to_node[Yid] = nid

    # Enrich trace with this info.
    nid_to_nodes = {t[0]: t for t in execution_trace}
    tmp = []
    for nid, l, Xs, Ys in execution_trace:
        if isinstance(l, keras.layers.InputLayer):
            # The nids that created or receive the tensors.
            Xs_nids = []  # Input layer does not receive.
            Ys_nids = [inputs_to_node[id(Y)] for Y in Ys]
            # The layers that created or receive the tensors.
            Xs_layers = []  # Input layer does not receive.
            Ys_layers = [[nid_to_nodes[Ynid][1] for Ynid in Ynids2]
                         for Ynids2 in Ys_nids]
        else:
            # The nids that created or receive the tensors.
            Xs_nids = [outputs_to_node.get(id(X), None) for X in Xs]
            Ys_nids = [inputs_to_node.get(id(Y), [None]) for Y in Ys]
            # The layers that created or receive the tensors.
            Xs_layers = [nid_to_nodes[Xnid][1]
                         for Xnid in Xs_nids if Xnid is not None]
            Ys_layers = [[nid_to_nodes[Ynid][1]
                          for Ynid in Ynids2 if Ynid is not None]
                         for Ynids2 in Ys_nids]

        entry = {
            "nid": nid,
            "layer": l,
            "Xs": Xs,
            "Ys": Ys,
            "Xs_nids": Xs_nids,
            "Ys_nids": Ys_nids,
            "Xs_layers": Xs_layers,
            "Ys_layers": Ys_layers,
        }
        tmp.append(entry)
    execution_trace = tmp

    if not keep_input_layers:
        execution_trace = [tmp
                           for tmp in execution_trace
                           if tmp["nid"] is not None]

    return execution_trace


def get_model_execution_graph(model, keep_input_layers=False):
    """
    Returns a dictionary representing the execution graph.
    Each key is the node's id as it is used by the reverse_model method.

    Each associated value contains a dictionary with the following items:

    * nid: the node id.
    * layer: the layer creating this node.
    * Xs: the input tensors (only valid if not in a nested container).
    * Ys: the output tensors (only valid if not in a nested container).
    * Xs_nids: the ids of the nodes creating the Xs.
    * Ys_nids: the ids of nodes using the according output tensor.
    * Xs_layers: the layer that created the accodring input tensor.
    * Ys_layers: the layers using the according output tensor.

    :param model: A kera model.
    :param keep_input_layers: Keep input layers.
    """
    trace = get_model_execution_trace(model,
                                      keep_input_layers=keep_input_layers,
                                      reapply_on_copied_layers=False)

    input_layers = [tmp for tmp in trace if tmp["nid"] is None]
    graph = {tmp["nid"]: tmp for tmp in trace}
    if keep_input_layers:
        graph[None] = input_layers

    return graph


def print_model_execution_graph(graph):
    """Pretty print of a model execution graph."""

    def nids_as_str(nids):
        return ", ".join(["%s" % nid for nid in nids])

    def print_node(node):
        print("  [NID: %4s] [Layer: %20s] "
              "[Inputs from: %20s] [Outputs to: %20s]" %
              (node["nid"],
               node["layer"].name,
               nids_as_str(node["Xs_nids"]),
               nids_as_str(node["Ys_nids"]),))

    if None in graph:
        print("Graph input layers:")
        for tmp in graph[None]:
            print_node(tmp)

    print("Graph nodes:")
    for nid in sorted([k for k in graph if k is not None]):
        if nid is None:
            continue
        print_node(graph[nid])


def get_bottleneck_nodes(inputs, outputs, execution_list):
    """
    Given an execution list this function returns all nodes that
    are a bottleneck in the network, i.e., "all information" must pass
    through this node.
    """

    forward_connections = {}
    for l, Xs, Ys in execution_list:
        if isinstance(l, keras.layers.InputLayer):
            # Special case, do nothing.
            continue

        for x in Xs:
            if x in forward_connections:
                forward_connections[x] += Ys
            else:
                forward_connections[x] = list(Ys)

    open_connections = {}
    for x in inputs:
        for fw_c in forward_connections[x]:
            open_connections[fw_c] = True

    ret = list()
    for l, Xs, Ys in execution_list:
        if isinstance(l, keras.layers.InputLayer):
            # Special case, do nothing.
            # Note: if a single input branches
            # this is not detected.
            continue

        for y in Ys:
            assert y in open_connections
            del open_connections[y]

        if len(open_connections) == 0:
            ret.append((l, (Xs, Ys)))

        for y in Ys:
            if y not in outputs:
                for fw_c in forward_connections[y]:
                    open_connections[fw_c] = True

    return ret


def get_bottleneck_tensors(inputs, outputs, execution_list):
    """
    Given an execution list this function returns all tensors that
    are a bottleneck in the network, i.e., "all information" must pass
    through this tensor.
    """

    nodes = get_bottleneck_nodes(inputs, outputs, execution_list)

    ret = list()
    for l, (Xs, Ys) in nodes:
        for tensor_list in (Xs, Ys):
            if len(tensor_list) == 1:
                tensor = tensor_list[0]
                if tensor not in ret:
                    ret.append(tensor)
            else:
                # TODO(albermax): put warning here?
                pass
    return ret

