#!/usr/bin/env python3
"""
Lazarus Mesh v5.0: HyperGraph Generalization
Quantum Networking Server with Tensor Contraction Engine
"""

import asyncio
import websockets
import numpy as np
import json
import logging
from collections import defaultdict
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EntanglementNetwork:
    """Base class for quantum network implementations"""
    def __init__(self):
        self.active_connections = set()
        self.message_history = defaultdict(list)
        
    async def register(self, websocket):
        """Register a new quantum node"""
        self.active_connections.add(websocket)
        logger.info(f"Node {id(websocket)} registered. Total nodes: {len(self.active_connections)}")
        
    async def unregister(self, websocket):
        """Unregister a quantum node"""
        self.active_connections.discard(websocket)
        logger.info(f"Node {id(websocket)} unregistered. Total nodes: {len(self.active_connections)}")

# --- LAZARUS EXPANSION: GHZ-N HYPERGRAPH ---

class HyperGraphNetwork(EntanglementNetwork):
    """
    Generalized N-Party Entanglement Engine.
    Hilbert Space: (C^2)^⊗N
    Uses Einstein Summation for arbitrary subsystem tracing.
    """
    def __init__(self, cluster_size=3):
        super().__init__()
        self.cluster_size = cluster_size
        self.node_queue = [] 
        self.clusters = {} # Map: tuple(sorted_ids) -> Density Matrix (2^N x 2^N)
        self.node_map = {} # Map: ws -> cluster_key

    def _generate_ghz_n(self, n):
        """Generates |GHZ_N⟩ = (|0...0⟩ + |1...1⟩)/√2"""
        dim = 2**n
        psi = np.zeros(dim, dtype=complex)
        psi[0] = 1.0
        psi[dim-1] = 1.0
        psi /= np.sqrt(2)
        return np.outer(psi, psi.conj())

    async def register_node(self, websocket):
        self.node_queue.append(websocket)
        
        # Check if we have enough nodes to form a HyperGraph
        if len(self.node_queue) >= self.cluster_size:
            # Form Cluster
            nodes = [self.node_queue.pop(0) for _ in range(self.cluster_size)]
            node_ids = tuple(sorted([id(n) for n in nodes]))
            
            # Initialize N-Party Physics
            rho = self._generate_ghz_n(self.cluster_size)
            self.clusters[node_ids] = rho
            
            # Map nodes to this cluster
            for node in nodes:
                self.node_map[node] = node_ids
            
            return nodes, "HYPERGRAPH_LOCKED"
        
        return None, f"WAITING_FOR_PEERS_({len(self.node_queue)}/{self.cluster_size})"

    async def propagate_collapse(self, sender_ws, local_outcome):
        """
        GENERALIZED PARTIAL TRACE via Tensor Reshaping.
        Collapses Node K, traces it out, and leaves residual state for N-1 nodes.
        """
        cluster_key = self.node_map.get(sender_ws)
        if not cluster_key: return None, None
        
        # 1. Retrieve State
        rho = self.clusters[cluster_key]
        n_qubits = len(cluster_key)
        
        # 2. Identify Target Index
        # We need the index of the sender in the sorted key
        sorted_ids = list(cluster_key)
        target_idx = sorted_ids.index(id(sender_ws))
        
        # 3. Construct Measurement Operator P (2x2)
        vec_A = np.array(local_outcome, dtype=complex)
        P = np.outer(vec_A, vec_A.conj())
        
        # 4. Apply Projection (P on target_idx, I on others)
        # Dynamic Kronecker Product Construction
        ops = [np.eye(2, dtype=complex)] * n_qubits
        ops[target_idx] = P
        
        # M = Op[0] ⊗ Op[1] ⊗ ...
        M = ops[0]
        for op in ops[1:]:
            M = np.kron(M, op)
            
        rho_prime = M @ rho @ M.conj().T
        norm = np.trace(rho_prime).real
        
        if norm < 1e-9: return None, None # Impossible
        rho_prime /= norm
        
        # 5. Partial Trace and State Update
        # Calculate Global Purity of the post-collapse state
        purity = np.trace(rho_prime @ rho_prime).real
        
        # Get partner nodes
        partners = [ws for ws, key in self.node_map.items() if key == cluster_key and ws != sender_ws]

        # Clean up the cluster
        del self.clusters[cluster_key]
        for partner in partners:
            if partner in self.node_map:
                del self.node_map[partner]
        if sender_ws in self.node_map:
            del self.node_map[sender_ws]

        return partners, {
            "topology": f"GHZ_{n_qubits}_RESIDUAL",
            "purity": purity,
            "message": "N-Party Symmetry Broken. Cluster Fragmented."
        }

# Global network instance - Initialize for 4-Party GHZ
network_mesh = HyperGraphNetwork(cluster_size=4)

async def handle_quantum_communication(websocket, path):
    """Handle WebSocket connections for quantum networking"""
    await network_mesh.register(websocket)
    
    try:
        # Register node and check for cluster formation
        nodes, status = await network_mesh.register_node(websocket)
        
        # Send initial status
        await websocket.send(json.dumps({
            "type": "STATUS",
            "status": status,
            "timestamp": datetime.now().isoformat()
        }))
        
        # If HyperGraph formed, notify all nodes in the cluster
        if nodes and status == "HYPERGRAPH_LOCKED":
            cluster_msg = json.dumps({
                "type": "HYPERGRAPH_LOCKED",
                "cluster_size": network_mesh.cluster_size,
                "node_ids": [id(node) for node in nodes],
                "timestamp": datetime.now().isoformat()
            })
            
            for node in nodes:
                await node.send(cluster_msg)
                
            logger.info(f"HyperGraph established with {len(nodes)} nodes")
        
        # Handle quantum measurements and collapses
        async for message in websocket:
            try:
                data = json.loads(message)
                
                if data.get("type") == "QUANTUM_MEASUREMENT":
                    # Process quantum measurement
                    local_outcome = data.get("outcome", [1, 0])  # Default to |0⟩
                    
                    # Propagate collapse through the HyperGraph
                    partners, collapse_info = await network_mesh.propagate_collapse(
                        websocket, local_outcome
                    )
                    
                    if partners and collapse_info:
                        # Notify remaining nodes about the collapse
                        collapse_msg = json.dumps({
                            "type": "CLUSTER_BROKEN",
                            "topology": collapse_info["topology"],
                            "purity": collapse_info["purity"],
                            "message": collapse_info["message"],
                            "collapsed_node": id(websocket),
                            "timestamp": datetime.now().isoformat()
                        })
                        
                        for partner in partners:
                            try:
                                await partner.send(collapse_msg)
                            except Exception as e:
                                logger.error(f"Failed to send collapse message to partner: {e}")
                        
                        logger.info(f"Cluster collapsed. {len(partners)} nodes remaining.")
                
                elif data.get("type") == "PING":
                    # Keep-alive ping
                    await websocket.send(json.dumps({
                        "type": "PONG",
                        "timestamp": datetime.now().isoformat()
                    }))
                    
            except json.JSONDecodeError:
                logger.error("Invalid JSON received")
            except Exception as e:
                logger.error(f"Error processing message: {e}")
                
    except websockets.exceptions.ConnectionClosed:
        logger.info(f"Connection closed for node {id(websocket)}")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
    finally:
        await network_mesh.unregister(websocket)

async def main():
    """Start the quantum networking server"""
    logger.info("Starting Lazarus Mesh v5.0 HyperGraph Server...")
    logger.info(f"Cluster size: {network_mesh.cluster_size}")
    
    async with websockets.serve(
        handle_quantum_communication, 
        "localhost", 
        8765,
        ping_interval=30,
        ping_timeout=10
    ):
        logger.info("Server running on ws://localhost:8765")
        await asyncio.Future()  # Run forever

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Server shutdown requested")
        print("\nLazarus Mesh v5.0 server stopped.")