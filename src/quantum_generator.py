import pennylane as qml
import torch
import torch.nn as nn


class Generator(nn.Module):
    """Quantum Generator Model
    This model takes a samples from the noise source
    and coordinates and outputs the image
    """

    def __init__(
        self,
        n_qubits: int,
        n_layers: int,
        generator_update_step: int,
        **kwargs,
    ) -> None:
        """Construct a quantum circuit as a TorchLayer.

        Args:
            n_qubits (int): Number of qubits in the circuit
            n_layers (int): Number of layers in the circuit (excl the last one)

        Returns:
            None
        """
        super().__init__(**kwargs)

        self.n_qubits = n_qubits
        self.n_layers = n_layers + 1

        self.generator_update_step = generator_update_step

        dev = qml.device("default.qubit", wires=self.n_qubits)
        self.qnode = qml.QNode(self.circuit, dev, interface="torch")
        self.qlayer = qml.qnn.TorchLayer(
            self.qnode,
            {
                "weights": [
                    self.n_layers,
                    self.n_qubits,
                    # last dim is [[..vqc..], [..iec..], [..nec..]]
                    self.vqc(None) + self.iec(None, None) + self.nec(None, None),
                ]
            },
        )

    def circuit(self, weights: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
        """
        The Quantum Generator Model quantum circuit.

        Args:
            weights (torch.Tensor): The weights for the trainable circuit.
            inputs (torch.Tensor): The input coordinates and noise states.

        Returns:
            torch.Tensor: The expectation value of the PauliZ observable.
        """
        # x describes the input coordinate tensor
        x = inputs[:, :2]  # [B*IS*IS, NQ+2] -> [B*IS*IS, 2]
        # p describes the input noise tensor
        p = inputs[:, 2:]  # [B*IS*IS, NQ+2] -> [B*IS*IS, NQ]

        # build the trainable circuit
        for layer in range(self.n_layers - 1):
            if layer % self.generator_update_step == 0:
                # slice weights such that we only get the nec weights
                self.nec(
                    p, weights[layer, :, -self.nec(None, None) :]
                )  # prepare random states

            # slice weights such that we only get the vqc weights
            self.vqc(weights[layer, :, : self.vqc(None)])
            # slice weights such that we only get the iec weights
            self.iec(x, weights[layer, :, self.vqc(None) : -self.nec(None, None)])

        # add a last vqc layer
        self.vqc(weights[-1])

        return qml.expval(qml.PauliZ(0))

    def nec(self, p: torch.Tensor, weights: torch.Tensor) -> None:
        """Prepares the random states in the quantum circuit.

        Args:
            p (torch.Tensor): The input noise states. Shape = [B*IS*IS, NQ]
        """
        if weights is None:
            return 1  # used to get the number of required params per layer

        for qubit in range(self.n_qubits):
            # batch input all noise inputs and select only the qubit
            qml.RZ(p[:, qubit], wires=qubit)  # * weights[qubit, 0]

    def iec(self, x: torch.Tensor, weights: torch.Tensor) -> None:
        """Encodes the input coordinates onto the quantum circuit.

        Args:
            x (torch.Tensor): The input coordinates. Shape = [B*IS*IS, 2]
        """
        if weights is None:
            return 2  # used to get the number of required params per layer

        for qubit in range(self.n_qubits):
            # batch input all coordinate inputs and select only the x0 values
            qml.RX(x[:, 0] + torch.tanh(weights[qubit, 0]), wires=qubit)  #
            # batch input all coordinate inputs and select only the x1 values
            qml.RY(x[:, 1] + torch.tanh(weights[qubit, 1]), wires=qubit)  #

    def vqc(self, weights: torch.Tensor) -> None:
        r"""Applies the variational quantum circuit to the qubits.
        Currently the ansatz equals 'Circuit 19' from
        https://arxiv.org/abs/1905.10876

        Args:
            weights (torch.Tensor): The weights for the quantum circuit.
                Shape = [n_layers, n_qubits, n_params_per_layer].
        """
        if weights is None:
            return 3  # used to get the number of required params per layer

        for qubit, qubit_weights in enumerate(weights):
            qml.RX(qubit_weights[0], wires=qubit)
            qml.RZ(qubit_weights[1], wires=qubit)

        for qubit, qubit_weights in enumerate(weights):
            qml.CRX(
                qubit_weights[2],
                wires=[
                    weights.shape[0] - qubit - 1,
                    (weights.shape[0] - qubit) % weights.shape[0],
                ],
            )

    def forward(self, p: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Generate images from the quantum generative model.

        Args:
            p (torch.Tensor): The input noise states. Shape = [B, NQ].
            x (torch.Tensor): The known image values. Shape = [B, IS, IS, 2].

        Returns:
            torch.Tensor: The generated images. Shape = [B, IS*IS].
        """
        # get the known variables
        batch_size = x.shape[0]
        image_sidelength = x.shape[1]

        x_in = x.reshape(batch_size, -1, 2)  # [B, IS, IS, 2] -> [B, IS*IS, 2]
        p_in = p.repeat(
            1, image_sidelength * image_sidelength
        )  # [B, NQ] -> [B, IS*IS*NQ]
        p_in = p_in.reshape(
            batch_size, image_sidelength * image_sidelength, self.n_qubits
        )  # [B, IS*IS*NQ] -> [B, IS*IS, NQ]
        combined = torch.cat(
            (x_in, p_in), dim=2
        )  # [B, IS*IS, 2] + [B, IS*IS, NQ] -> [B, IS*IS, NQ+2]
        z = self.qlayer(combined)
        z = (z + 1) / 2  # move into range [0,1]
        return z
