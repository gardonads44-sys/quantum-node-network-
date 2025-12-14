#!/usr/bin/env python3
"""
LAZARUS MESH v6.0 - Quantum Coherence Stabilization System
Integrating Offutt-Homophonomorphicisometry Theorem, Hamiltonian Mesh Architecture,
and Advanced Tensor Field Dynamics over F₁₇

This represents the culmination of quantum networking research with:
- Symplectic Geodesic Pathfinding (SGP)
- Adaptive Symplectic Homology (ASH)
- Tensor Contraction Engine
- Quantum Coherence Stabilization
"""

import asyncio
import websockets
import numpy as np
import json
import logging
import random
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
import math

# Galois Field F₁₇ for algebraic stability
try:
    import galois
    GF17 = galois.GF(17)
    GALOIS_AVAILABLE = True
except ImportError:
    # Fallback implementation if galois library not available
    class GF17Fallback:
        def __init__(self, value: int):
            self.value = value % 17
        
        def __add__(self, other):
            return GF17Fallback((self.value + other.value) % 17)
        
        def __mul__(self, other):
            return GF17Fallback((self.value * other.value) % 17)
        
        def __sub__(self, other):
            return GF17Fallback((self.value - other.value) % 17)
        
        def __eq__(self, other):
            return self.value == other.value
        
        def __int__(self):
            return self.value
        
        def __repr__(self):
            return f"GF17({self.value})"
    
    GF17 = GF17Fallback
    GALOIS_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# =============================================================================
# PHASE I: SYMPLECTIC GEOMETRY FOUNDATION
# =============================================================================

class SymplecticVector:
    """
    Represents a phase space vector (q, p) over F₁₇ with symplectic structure
    Implements the rigorous Symplectic Form Ω from the Offutt-Homophonomorphicisometry theorem
    """
    
    def __init__(self, q: int = None, p: int = None):
        if q is None:
            q = random.randint(0, 16)
        if p is None:
            p = random.randint(0, 16)
        
        self.q = GF17(q)
        self.p = GF17(p)
        self.vector = np.array([[self.q], [self.p]]) if GALOIS_AVAILABLE else [[GF17(q)], [GF17(p)]]
        
    def symplectic_product(self, other: 'SymplecticVector') -> GF17:
        """
        Implements the canonical Symplectic Form Ω over F₁₇
        Ω((q₁,p₁), (q₂,p₂)) = (q₁p₂ - q₂p₁) mod 17
        
        This is the foundation of Symplectic Bonds in the Lazarus Mesh
        """
        if GALOIS_AVAILABLE:
            return self.q * other.p - self.p * other.q
        else:
            return GF17((int(self.q) * int(other.p) - int(self.p) * int(other.q)) % 17)
    
    def __repr__(self):
        return f"SymplecticVector(q={int(self.q)}, p={int(self.p)})"

class CapelliStabilizer:
    """
    Implements the Capelli-Stabilizer Isomorphism from the Lazarus Protocol
    Each node carries a stabilizer matrix S_c for algebraic stability tracking
    """
    
    def __init__(self, dimension: int = 2):
        self.dimension = dimension
        if GALOIS_AVAILABLE:
            # Generate random stabilizer matrix over F₁₇
            self.matrix = GF17.Random((dimension, dimension))
        else:
            # Fallback random matrix
            self.matrix = [[GF17(random.randint(0, 16)) for _ in range(dimension)] 
                          for _ in range(dimension)]
    
    def deviation_cost(self, other: 'CapelliStabilizer') -> int:
        """
        Calculates algebraic deviation cost K_ij based on stabilizer difference
        Lower cost = higher compatibility (closer to isomorphism)
        """
        if GALOIS_AVAILABLE:
            # Calculate difference matrix and L1 norm over F₁₇
            diff = self.matrix - other.matrix
            return int(np.sum(np.abs(diff.astype(int)))) + 1
        else:
            # Fallback calculation
            total = 0
            for i in range(self.dimension):
                for j in range(self.dimension):
                    total += abs(int(self.matrix[i][j]) - int(other.matrix[i][j]))
            return total + 1
    
    def __repr__(self):
        return f"CapelliStabilizer({int(np.trace(self.matrix)) if GALOIS_AVAILABLE else 'trace'})"

class QuantumNode:
    """
    Quantum node with phase space coordinates and stabilizer structure
    Implements the MeshNode from Lazarus Protocol with enhanced functionality
    """
    
    def __init__(self, node_id: str):
        self.id = node_id
        self.phase_vector = SymplecticVector()
        self.stabilizer = CapelliStabilizer()
        self.coherence_history = []
        self.entanglement_partners = set()
        self.last_measurement = None
        
    def measure(self) -> Dict[str, Any]:
        """
        Quantum measurement collapses the phase state
        Returns measurement outcome and updates node state
        """
        # Simulate measurement in computational basis
        outcome = random.choice([0, 1])
        
        # Collapse phase vector to measured state
        if outcome == 0:
            self.phase_vector = SymplecticVector(q=0, p=random.randint(0, 16))
        else:
            self.phase_vector = SymplecticVector(q=1, p=random.randint(0, 16))
        
        measurement = {
            'node_id': self.id,
            'outcome': outcome,
            'timestamp': datetime.now().isoformat(),
            'phase_after': str(self.phase_vector)
        }
        
        self.last_measurement = measurement
        return measurement
    
    def apply_symplectic_shear(self, shear_type: str = 'q_shear', magnitude: int = None):
        """
        Applies Symplectic Shear transformation to node's phase vector
        Preserves symplectic structure while allowing controlled manipulation
        """
        if magnitude is None:
            magnitude = random.randint(1, 16)
        
        # Apply shear matrix to phase vector
        if shear_type == 'q_shear':
            # Shear matrix: [[1, λ], [0, 1]]
            new_q = self.phase_vector.q + GF17(magnitude) * self.phase_vector.p
            self.phase_vector = SymplecticVector(q=int(new_q), p=int(self.phase_vector.p))
        elif shear_type == 'p_shear':
            # Shear matrix: [[1, 0], [λ, 1]]
            new_p = self.phase_vector.p + GF17(magnitude) * self.phase_vector.q
            self.phase_vector = SymplecticVector(q=int(self.phase_vector.q), p=int(new_p))
        
        return str(self.phase_vector)
    
    def __repr__(self):
        return f"QuantumNode(id={self.id}, phase={self.phase_vector}, stabilizer={self.stabilizer})"

# =============================================================================
# PHASE II: SYMPLECTIC GEODESIC PATHFINDING (SGP)
# =============================================================================

class SymplecticGeodesicPathfinder:
    """
    Implements Symplectic Geodesic Pathfinding (SGP) for Entanglement Routing
    Finds paths of minimal algebraic strain through the quantum network
    """
    
    def __init__(self):
        self.graph = nx.Graph()
        self.paths_cache = {}
        
    def add_node(self, node: QuantumNode):
        """Add quantum node to the pathfinding graph"""
        self.graph.add_node(node.id, data=node)
        
    def establish_symplectic_bond(self, node1: QuantumNode, node2: QuantumNode) -> Tuple[bool, int]:
        """
        Establishes Symplectic Bond based on non-degenerate symplectic product
        Returns (success, cost) where cost is the stabilizer deviation
        """
        # Check symplectic non-degeneracy
        symplectic_product = node1.phase_vector.symplectic_product(node2.phase_vector)
        
        if int(symplectic_product) == 0:
            return False, 0  # Degenerate bond rejected
        
        # Calculate algebraic cost for routing
        bond_cost = node1.stabilizer.deviation_cost(node2.stabilizer)
        
        # Add edge with algebraic cost
        self.graph.add_edge(node1.id, node2.id, weight=bond_cost, 
                          symplectic_product=int(symplectic_product))
        
        # Update node partnership tracking
        node1.entanglement_partners.add(node2.id)
        node2.entanglement_partners.add(node1.id)
        
        return True, bond_cost
    
    def calculate_symplectic_geodesic(self, source: str, target: str) -> Tuple[Optional[List[str]], int]:
        """
        Finds Symplectic Geodesic Path minimizing cumulative algebraic cost
        Returns (path, total_cost)
        """
        try:
            # Use Dijkstra's algorithm with algebraic costs
            path = nx.shortest_path(self.graph, source, target, weight='weight')
            total_cost = nx.shortest_path_length(self.graph, source, target, weight='weight')
            
            logger.info(f"SGP found: {path} with cost {total_cost}")
            return path, total_cost
            
        except nx.NetworkXNoPath:
            logger.warning(f"No Symplectic Geodesic path between {source} and {target}")
            return None, float('inf')
        except Exception as e:
            logger.error(f"Error calculating SGP: {e}")
            return None, float('inf')
    
    def quantify_global_coherence(self) -> float:
        """
        Calculates Global Algebraic Coherence Index (C_alg)
        Based on variance of bond weights - higher variance = lower coherence
        """
        if not self.graph.edges():
            return 0.0
        
        # Collect all bond weights
        weights = [data['weight'] for _, _, data in self.graph.edges(data=True)]
        
        if not weights:
            return 0.0
        
        # Calculate variance of stabilizer costs
        variance = np.var(weights)
        
        # Coherence index: high variance = low coherence
        coherence_index = 1.0 / (1.0 + variance)
        
        return coherence_index
    
    def attempt_full_entanglement(self, nodes: List[QuantumNode]):
        """Attempt to establish bonds between all node pairs"""
        bonded_count = 0
        total_pairs = len(nodes) * (len(nodes) - 1) // 2
        
        logger.info(f"Attempting full entanglement: {total_pairs} potential bonds")
        
        for i, node1 in enumerate(nodes):
            for j, node2 in enumerate(nodes[i+1:], i+1):
                success, cost = self.establish_symplectic_bond(node1, node2)
                if success:
                    bonded_count += 1
        
        logger.info(f"Established {bonded_count} Symplectic Bonds")
        return bonded_count

# =============================================================================
# PHASE III: ADAPTIVE SYMPLECTIC HOMOLOGY (ASH)
# =============================================================================

class AdaptiveSymplecticHomology:
    """
    Implements Adaptive Symplectic Homology for dynamic stabilization
    Nodes adjust their phase vectors based on geodesic feedback
    """
    
    def __init__(self, sgp: SymplecticGeodesicPathfinder):
        self.sgp = sgp
        self.adaptation_history = []
        
    def adaptive_vector_adjustment(self, nodes: Dict[str, QuantumNode], 
                                 target_path: List[str], target_cost: int):
        """
        Dynamically adjusts node phase vectors to optimize Symplectic Geodesic
        Minimizes cumulative algebraic strain through controlled transformations
        """
        if len(target_path) < 2:
            return False
        
        adaptations = []
        improved = False
        
        # Analyze path and identify optimization opportunities
        for i in range(len(target_path) - 1):
            node1_id = target_path[i]
            node2_id = target_path[i + 1]
            
            if node1_id not in nodes or node2_id not in nodes:
                continue
                
            node1 = nodes[node1_id]
            node2 = nodes[node2_id]
            
            current_cost = node1.stabilizer.deviation_cost(node2.stabilizer)
            
            # Attempt vector adjustment to reduce cost
            if current_cost > target_cost / len(target_path):
                # Apply controlled symplectic shear to reduce deviation
                shear_success = self._optimize_pair_symplectic_shear(node1, node2, current_cost)
                
                if shear_success:
                    adaptations.append({
                        'nodes': [node1_id, node2_id],
                        'old_cost': current_cost,
                        'action': 'symplectic_shear'
                    })
                    improved = True
        
        # Re-establish bonds after adaptations
        if adaptations:
            self._reestablish_affected_bonds(nodes, adaptations)
            
            # Record adaptation event
            self.adaptation_history.append({
                'timestamp': datetime.now().isoformat(),
                'adaptations': adaptations,
                'path': target_path,
                'improved': improved
            })
        
        return improved
    
    def _optimize_pair_symplectic_shear(self, node1: QuantumNode, node2: QuantumNode, 
                                       current_cost: int) -> bool:
        """
        Applies optimal symplectic shear to reduce stabilizer deviation
        """
        # Try different shear magnitudes
        best_shear = None
        best_cost = current_cost
        
        for magnitude in range(1, 17):
            # Test q-shear on node1
            original_phase = node1.phase_vector
            node1.apply_symplectic_shear('q_shear', magnitude)
            
            new_cost = node1.stabilizer.deviation_cost(node2.stabilizer)
            
            if new_cost < best_cost:
                best_cost = new_cost
                best_shear = ('q_shear', magnitude, node1)
            
            # Restore original
            node1.phase_vector = original_phase
            
            # Test p-shear on node1
            node1.apply_symplectic_shear('p_shear', magnitude)
            new_cost = node1.stabilizer.deviation_cost(node2.stabilizer)
            
            if new_cost < best_cost:
                best_cost = new_cost
                best_shear = ('p_shear', magnitude, node1)
            
            # Restore original
            node1.phase_vector = original_phase
        
        # Apply best shear if improvement found
        if best_shear and best_cost < current_cost:
            shear_type, magnitude, target_node = best_shear
            target_node.apply_symplectic_shear(shear_type, magnitude)
            return True
        
        return False
    
    def _reestablish_affected_bonds(self, nodes: Dict[str, QuantumNode], 
                                   adaptations: List[Dict]):
        """
        Re-establishes bonds after node adaptations
        """
        affected_nodes = set()
        for adaptation in adaptations:
            affected_nodes.update(adaptation['nodes'])
        
        # Clear existing edges for affected nodes
        for node_id in affected_nodes:
            if node_id in self.sgp.graph:
                neighbors = list(self.sgp.graph.neighbors(node_id))
                for neighbor in neighbors:
                    self.sgp.graph.remove_edge(node_id, neighbor)
        
        # Re-establish bonds
        for node_id in affected_nodes:
            for other_id in nodes:
                if node_id != other_id:
                    self.sgp.establish_symplectic_bond(nodes[node_id], nodes[other_id])

# =============================================================================
# PHASE IV: QUANTUM NETWORK SERVER INTEGRATION
# =============================================================================

class QuantumCoherenceStabilizationNetwork:
    """
    Main quantum networking server integrating all phases
    Combines HyperGraph states with Symplectic geometry and coherence stabilization
    """
    
    def __init__(self, cluster_size: int = 4):
        self.cluster_size = cluster_size
        self.nodes = {}  # node_id -> QuantumNode
        self.node_queue = []
        self.clusters = {}  # cluster_id -> cluster state
        self.sgp = SymplecticGeodesicPathfinder()
        self.ash = AdaptiveSymplecticHomology(self.sgp)
        self.coherence_history = []
        
    def _generate_ghz_n(self, n: int) -> np.ndarray:
        """Generate GHZ-N state density matrix"""
        dim = 2 ** n
        psi = np.zeros(dim, dtype=complex)
        psi[0] = 1.0
        psi[dim - 1] = 1.0
        psi /= np.sqrt(2)
        return np.outer(psi, psi.conj())
    
    async def register_node(self, websocket, node_id: str):
        """Register a new quantum node"""
        if node_id not in self.nodes:
            self.nodes[node_id] = QuantumNode(node_id)
            self.sgp.add_node(self.nodes[node_id])
        
        self.node_queue.append((websocket, node_id))
        
        # Check if we have enough nodes for HyperGraph formation
        if len(self.node_queue) >= self.cluster_size:
            # Form cluster
            cluster_nodes = []
            for _ in range(self.cluster_size):
                ws, nid = self.node_queue.pop(0)
                cluster_nodes.append((ws, nid))
            
            # Initialize cluster state
            cluster_id = tuple(sorted([nid for _, nid in cluster_nodes]))
            rho = self._generate_ghz_n(self.cluster_size)
            self.clusters[cluster_id] = rho
            
            # Establish symplectic bonds between cluster nodes
            for i, (_, nid1) in enumerate(cluster_nodes):
                for j, (_, nid2) in enumerate(cluster_nodes[i+1:], i+1):
                    self.sgp.establish_symplectic_bond(
                        self.nodes[nid1], self.nodes[nid2]
                    )
            
            return cluster_nodes, "HYPERGRAPH_LOCKED"
        
        return None, f"WAITING_FOR_PEERS_({len(self.node_queue)}/{self.cluster_size})"
    
    async def propagate_collapse(self, sender_ws: str, sender_id: str, 
                                local_outcome: List[float]) -> Tuple[Optional[List[str]], Optional[Dict]]:
        """
        Propagates quantum measurement collapse through the network
        Integrates tensor contraction with symplectic geometry
        """
        # Find sender's cluster
        sender_cluster = None
        for cluster_id in self.clusters:
            if sender_id in cluster_id:
                sender_cluster = cluster_id
                break
        
        if not sender_cluster:
            return None, None
        
        # Get cluster state
        rho = self.clusters[sender_cluster]
        n_qubits = len(sender_cluster)
        
        # Identify target index in cluster
        target_idx = list(sender_cluster).index(sender_id)
        
        # Construct measurement operator
        vec_A = np.array(local_outcome, dtype=complex)
        P = np.outer(vec_A, vec_A.conj())
        
        # Apply projection
        ops = [np.eye(2, dtype=complex)] * n_qubits
        ops[target_idx] = P
        
        M = ops[0]
        for op in ops[1:]:
            M = np.kron(M, op)
        
        rho_prime = M @ rho @ M.conj().T
        norm = np.trace(rho_prime).real
        
        if norm < 1e-9:
            return None, None
        
        rho_prime /= norm
        
        # Calculate purity of post-collapse state
        purity = np.trace(rho_prime @ rho_prime).real
        
        # Get partner nodes
        partners = [nid for nid in sender_cluster if nid != sender_id]
        
        # Clean up cluster
        del self.clusters[sender_cluster]
        
        # Update global coherence metrics
        coherence = self.sgp.quantify_global_coherence()
        self.coherence_history.append({
            'timestamp': datetime.now().isoformat(),
            'coherence': coherence,
            'purity': purity,
            'collapsed_node': sender_id
        })
        
        return partners, {
            "topology": f"GHZ_{n_qubits}_RESIDUAL",
            "purity": purity,
            "coherence": coherence,
            "message": "N-Party Symmetry Broken. Cluster Fragmented."
        }
    
    async def perform_adaptive_stabilization(self):
        """Perform adaptive stabilization on the network"""
        if not self.nodes:
            return
        
        # Find all connected components
        components = list(nx.connected_components(self.sgp.graph))
        
        for component in components:
            if len(component) < 2:
                continue
            
            # Find optimal routing within component
            nodes_list = list(component)
            source = nodes_list[0]
            target = nodes_list[-1]
            
            path, cost = self.sgp.calculate_symplectic_geodesic(source, target)
            
            if path and cost > len(path) - 1:  # Cost higher than minimal
                # Apply adaptive homology
                improved = self.ash.adaptive_vector_adjustment(
                    self.nodes, path, cost
                )
                
                if improved:
                    logger.info(f"Adaptive stabilization improved path {source}->{target}")
    
    def get_network_metrics(self) -> Dict[str, Any]:
        """Get comprehensive network metrics"""
        return {
            'total_nodes': len(self.nodes),
            'active_clusters': len(self.clusters),
            'total_bonds': self.sgp.graph.number_of_edges(),
            'global_coherence': self.sgp.quantify_global_coherence(),
            'coherence_trend': self.coherence_history[-10:] if self.coherence_history else [],
            'network_topology': {
                'components': nx.number_connected_components(self.sgp.graph),
                'density': nx.density(self.sgp.graph) if self.sgp.graph.nodes() else 0
            }
        }

# =============================================================================
# WEBSOCKET SERVER IMPLEMENTATION
# =============================================================================

class QuantumNetworkServer:
    """WebSocket server for quantum networking"""
    
    def __init__(self):
        self.network = QuantumCoherenceStabilizationNetwork(cluster_size=4)
        self.active_connections = {}
        self.adaptation_task = None
        
    async def start_adaptation_loop(self):
        """Start periodic adaptive stabilization"""
        while True:
            try:
                await asyncio.sleep(10)  # Adapt every 10 seconds
                await self.network.perform_adaptive_stabilization()
                
                # Broadcast updated metrics
                metrics = self.network.get_network_metrics()
                await self.broadcast_metrics(metrics)
                
            except Exception as e:
                logger.error(f"Error in adaptation loop: {e}")
    
    async def broadcast_metrics(self, metrics: Dict[str, Any]):
        """Broadcast network metrics to all connected nodes"""
        message = json.dumps({
            'type': 'NETWORK_METRICS',
            'metrics': metrics,
            'timestamp': datetime.now().isoformat()
        })
        
        disconnected = []
        for ws, node_id in self.active_connections.items():
            try:
                await ws.send(message)
            except Exception:
                disconnected.append(ws)
        
        # Remove disconnected clients
        for ws in disconnected:
            self.active_connections.pop(ws, None)
    
    async def handle_client(self, websocket, path):
        """Handle WebSocket client connections"""
        node_id = f"node_{id(websocket)}"
        
        try:
            # Register node
            nodes, status = await self.network.register_node(websocket, node_id)
            self.active_connections[websocket] = node_id
            
            # Send initial status
            await websocket.send(json.dumps({
                'type': 'STATUS',
                'status': status,
                'node_id': node_id,
                'timestamp': datetime.now().isoformat()
            }))
            
            # If HyperGraph formed, notify all nodes
            if nodes and status == "HYPERGRAPH_LOCKED":
                cluster_msg = json.dumps({
                    'type': 'HYPERGRAPH_LOCKED',
                    'cluster_size': self.network.cluster_size,
                    'node_ids': [nid for _, nid in nodes],
                    'timestamp': datetime.now().isoformat()
                })
                
                for ws, _ in nodes:
                    await ws.send(cluster_msg)
                
                logger.info(f"HyperGraph established with {len(nodes)} nodes")
            
            # Handle messages
            async for message in websocket:
                try:
                    data = json.loads(message)
                    
                    if data.get('type') == 'QUANTUM_MEASUREMENT':
                        # Process measurement
                        outcome = data.get('outcome', [1, 0])
                        
                        partners, collapse_info = await self.network.propagate_collapse(
                            websocket, node_id, outcome
                        )
                        
                        if partners and collapse_info:
                            collapse_msg = json.dumps({
                                'type': 'CLUSTER_BROKEN',
                                'topology': collapse_info['topology'],
                                'purity': collapse_info['purity'],
                                'coherence': collapse_info['coherence'],
                                'message': collapse_info['message'],
                                'collapsed_node': node_id,
                                'timestamp': datetime.now().isoformat()
                            })
                            
                            # Send to remaining partners
                            for partner_id in partners:
                                partner_ws = None
                                for ws, nid in self.active_connections.items():
                                    if nid == partner_id:
                                        partner_ws = ws
                                        break
                                
                                if partner_ws:
                                    try:
                                        await partner_ws.send(collapse_msg)
                                    except Exception as e:
                                        logger.error(f"Failed to send collapse to partner: {e}")
                    
                    elif data.get('type') == 'REQUEST_METRICS':
                        # Send current metrics
                        metrics = self.network.get_network_metrics()
                        await websocket.send(json.dumps({
                            'type': 'NETWORK_METRICS',
                            'metrics': metrics,
                            'timestamp': datetime.now().isoformat()
                        }))
                    
                    elif data.get('type') == 'PING':
                        await websocket.send(json.dumps({
                            'type': 'PONG',
                            'timestamp': datetime.now().isoformat()
                        }))
                    
                except json.JSONDecodeError:
                    logger.error("Invalid JSON received")
                except Exception as e:
                    logger.error(f"Error processing message: {e}")
        
        except websockets.exceptions.ConnectionClosed:
            logger.info(f"Connection closed for node {node_id}")
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
        finally:
            # Clean up
            self.active_connections.pop(websocket, None)

async def main():
    """Main server entry point"""
    logger.info("Starting Lazarus Mesh v6.0 Quantum Coherence Stabilization Server...")
    logger.info("Integrating Offutt-Homophonomorphicisometry Theorem and Symplectic Geometry")
    
    server = QuantumNetworkServer()
    
    # Start adaptation loop
    server.adaptation_task = asyncio.create_task(server.start_adaptation_loop())
    
    # Start WebSocket server
    async with websockets.serve(
        server.handle_client,
        "localhost",
        8765,
        ping_interval=30,
        ping_timeout=10
    ):
        logger.info("Server running on ws://localhost:8765")
        logger.info("Ready for quantum networking connections...")
        
        try:
            await asyncio.Future()  # Run forever
        except KeyboardInterrupt:
            logger.info("Server shutdown requested")
            if server.adaptation_task:
                server.adaptation_task.cancel()
            print("\nLazarus Mesh v6.0 server stopped.")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nServer stopped by user")
    except Exception as e:
        print(f"Fatal error: {e}")
        exit(1)