from pytket.extensions.cirq import cirq_to_tk, tk_to_cirq
from pytket.passes import (  # type: ignore
    BasePass,
    auto_rebase_pass,
    RemoveRedundancies,
    SequencePass,
    SynthesiseTket,
    CXMappingPass,
    DecomposeBoxes,
    FullPeepholeOptimise,
    CliffordSimp,
    SimplifyInitial,
)
from pytket.architecture import Architecture
from pytket.placement import NoiseAwarePlacement
from pytket.extensions.cirq import (
    process_characterisation,
    get_avg_characterisation,
)
from pytket import OpType, Circuit
from cirq import AbstractCircuit
from cirq.devices import Device

from networkx.linalg.graphmatrix import adjacency_matrix

def rebase_pass():
    return auto_rebase_pass(
            {OpType.CX, OpType.X, OpType.SX, OpType.Rz},
        )

def coupling_map_pass(backend, passlist: list, averaged_errors: dict):
    coupling_map = adjacency_matrix(backend.metadata.nx_graph)
    arch = Architecture(coupling_map.todense())
    passlist.append(
        CXMappingPass(
            arch,
            NoiseAwarePlacement(
                arch,
                averaged_errors["node_errors"],
                averaged_errors["edge_errors"],
                averaged_errors["readout_errors"],
            ),
            directed_cx=False,
        )
    )

def high_optimisation(circuit: AbstractCircuit, backend: Device) -> list[AbstractCircuit]:
    """Perform thourough but generic optimisation using 
    TKET optimisation tool.

    :param circuit: Circuit to be optimised.
    :type circuit: QuantumCircuit
    :param backend: Backend which circuit should be optimised to.
    :type backend: BaseBackend
    :return: Optimised circuit.
    :rtype: list[QuantumCircuit]
    """
    print("  ... performing high TKET optimisation.")

    # Obtain device data for noise aware placement.
    characterisation = process_characterisation(backend)
    averaged_errors = get_avg_characterisation(characterisation)
    
    # Initialise pass list and perform thorough optimisation.
    passlist = [DecomposeBoxes()]
    passlist.append(FullPeepholeOptimise())

    # Add noise aware placement and routing to pass list.
    if "device" in str(type(backend)):
        coupling_map = adjacency_matrix(backend.metadata.nx_graph)
        arch = Architecture(coupling_map.todense())
        passlist.append(
            CXMappingPass(
                arch,
                NoiseAwarePlacement(
                    arch,
                    averaged_errors["node_errors"],
                    averaged_errors["edge_errors"],
                    averaged_errors["readout_errors"],
                ),
                directed_cx=False,
            )
        )

    # Perform coupling map safe optimisation.
    passlist.extend([CliffordSimp(False), SynthesiseTket()])

    # Rebase to backend gate set and perform basic optimisation
    passlist.append(rebase_pass())
    passlist.append(RemoveRedundancies())
    passlist.append(
        SimplifyInitial(allow_classical=False, create_all_qubits=True)
    )

    # Optimise circuit using constructed pass list
    tk_circuit = cirq_to_tk(circuit)
    SequencePass(passlist).apply(tk_circuit)
    circuit = tk_to_cirq(tk_circuit)

    return [circuit]

def quick_optimisation(circuit: AbstractCircuit, backend: Device) -> list[AbstractCircuit]:
    """Perform basic compilation to build valid circuit for backend.

    :param circuit: Circuit to be compiled.
    :type circuit: QuantumCircuit
    :param backend: Backend to compile to.
    :type backend: BaseBackend
    :return: Compiled circuit.
    :rtype: list[QuantumCircuit]
    """
    print("  ... performing quick TKET optimisation.")

    # Obtain device data for noise aware placement.
    characterisation = process_characterisation(backend)
    averaged_errors = get_avg_characterisation(characterisation)
    
    # Initialise pass list
    passlist = [DecomposeBoxes()]

    # Add noise aware placement and routing to pass list.
    if "device" in str(type(backend)):
        coupling_map = adjacency_matrix(backend.metadata.nx_graph)
        arch = Architecture(coupling_map.todense())
        passlist.append(
            CXMappingPass(
                arch,
                NoiseAwarePlacement(
                    arch,
                    averaged_errors["node_errors"],
                    averaged_errors["edge_errors"],
                    averaged_errors["readout_errors"],
                ),
                directed_cx=False,
            )
        )

    # Rebase to backend gate set and perform basic optimisation
    passlist.append(rebase_pass())

    # Optimise circuit using constructed pass list
    tk_circuit = cirq_to_tk(circuit)
    SequencePass(passlist).apply(tk_circuit)
    circuit = tk_to_cirq(tk_circuit)

    return [circuit]