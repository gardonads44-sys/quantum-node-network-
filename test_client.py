#!/usr/bin/env python3
"""
Lazarus Mesh v5.0 - Test Client for 4-Party GHZ State Simulation
Simulates multiple quantum nodes connecting to the HyperGraph network
"""

import asyncio
import websockets
import json
import numpy as np
import random
import logging
from datetime import datetime
from typing import List, Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class QuantumNode:
    """Simulates a single quantum node in the HyperGraph network"""
    
    def __init__(self, node_id: int, server_uri: str = "ws://localhost:8765"):
        self.node_id = node_id
        self.server_uri = server_uri
        self.websocket = None
        self.in_hypergraph = False
        self.cluster_nodes = []
        self.measurement_history = []
        
    async def connect(self):
        """Connect to the quantum network server"""
        try:
            self.websocket = await websockets.connect(self.server_uri)
            logger.info(f"Node {self.node_id}: Connected to {self.server_uri}")
            
            # Start listening for messages
            await self.listen()
            
        except Exception as e:
            logger.error(f"Node {self.node_id}: Failed to connect - {e}")
            raise
    
    async def disconnect(self):
        """Disconnect from the quantum network"""
        if self.websocket:
            await self.websocket.close()
            self.websocket = None
            self.in_hypergraph = False
            logger.info(f"Node {self.node_id}: Disconnected")
    
    async def listen(self):
        """Listen for messages from the server"""
        try:
            async for message in self.websocket:
                data = json.loads(message)
                await self.handle_message(data)
        except websockets.exceptions.ConnectionClosed:
            logger.info(f"Node {self.node_id}: Connection closed")
        except Exception as e:
            logger.error(f"Node {self.node_id}: Error in listener - {e}")
    
    async def handle_message(self, data: Dict[str, Any]):
        """Handle incoming messages from the server"""
        logger.info(f"Node {self.node_id}: Received {data.get('type', 'UNKNOWN')}")
        
        if data.get('type') == 'HYPERGRAPH_LOCKED':
            self.in_hypergraph = True
            self.cluster_nodes = data.get('node_ids', [])
            cluster_size = data.get('cluster_size', 4)
            
            logger.info(f"Node {self.node_id}: Joined HyperGraph cluster (size: {cluster_size})")
            logger.info(f"Node {self.node_id}: Cluster nodes: {self.cluster_nodes}")
            
            # Simulate random measurement after joining cluster
            await asyncio.sleep(random.uniform(1, 5))
            await self.perform_measurement()
            
        elif data.get('type') == 'CLUSTER_BROKEN':
            logger.info(f"Node {self.node_id}: Cluster broken - {data.get('message', 'N/A')}")
            logger.info(f"Node {self.node_id}: Topology: {data.get('topology', 'N/A')}")
            logger.info(f"Node {self.node_id}: State purity: {data.get('purity', 'N/A')}")
            
            self.in_hypergraph = False
            self.cluster_nodes = []
            
            # Record measurement in history
            self.measurement_history.append({
                'timestamp': datetime.now().isoformat(),
                'topology': data.get('topology'),
                'purity': data.get('purity'),
                'message': data.get('message')
            })
    
    async def perform_measurement(self):
        """Perform a quantum measurement and collapse the wavefunction"""
        if not self.in_hypergraph or not self.websocket:
            logger.warning(f"Node {self.node_id}: Not in HyperGraph, skipping measurement")
            return
        
        # Simulate quantum measurement outcome
        # Randomly choose between |0⟩ and |1⟩ states
        outcome = [1, 0] if random.random() < 0.5 else [0, 1]
        basis = "Z" if random.random() < 0.7 else "X"  # Usually measure in Z basis
        
        measurement_data = {
            'type': 'QUANTUM_MEASUREMENT',
            'outcome': outcome,
            'basis': basis,
            'timestamp': datetime.now().isoformat(),
            'node_id': self.node_id
        }
        
        await self.websocket.send(json.dumps(measurement_data))
        logger.info(f"Node {self.node_id}: Performed measurement in {basis} basis")
        
        # Record this measurement
        self.measurement_history.append(measurement_data)

class QuantumNetworkSimulator:
    """Simulates multiple quantum nodes in the HyperGraph network"""
    
    def __init__(self, num_nodes: int = 4, server_uri: str = "ws://localhost:8765"):
        self.num_nodes = num_nodes
        self.server_uri = server_uri
        self.nodes: List[QuantumNode] = []
        self.simulation_results = []
        
    async def initialize_nodes(self):
        """Initialize all quantum nodes"""
        logger.info(f"Initializing {self.num_nodes} quantum nodes...")
        
        for i in range(self.num_nodes):
            node = QuantumNode(node_id=i, server_uri=self.server_uri)
            self.nodes.append(node)
            
            # Stagger connections to simulate real-world timing
            await asyncio.sleep(random.uniform(0.1, 1.0))
            
            try:
                # Start node connection (non-blocking)
                asyncio.create_task(node.connect())
            except Exception as e:
                logger.error(f"Failed to start node {i}: {e}")
    
    async def run_simulation(self, duration: int = 30):
        """Run the simulation for a specified duration"""
        logger.info(f"Starting simulation for {duration} seconds...")
        
        await self.initialize_nodes()
        
        # Wait for the simulation to run
        await asyncio.sleep(duration)
        
        # Collect results
        await self.collect_results()
        
        # Disconnect all nodes
        await self.shutdown()
        
    async def collect_results(self):
        """Collect simulation results from all nodes"""
        logger.info("Collecting simulation results...")
        
        total_measurements = 0
        hypergraph_joins = 0
        cluster_breaks = 0
        
        for node in self.nodes:
            total_measurements += len(node.measurement_history)
            if any('HYPERGRAPH_LOCKED' in str(m) for m in node.measurement_history):
                hypergraph_joins += 1
            if any('CLUSTER_BROKEN' in str(m) for m in node.measurement_history):
                cluster_breaks += 1
        
        self.simulation_results = {
            'total_nodes': len(self.nodes),
            'total_measurements': total_measurements,
            'hypergraph_joins': hypergraph_joins,
            'cluster_breaks': cluster_breaks,
            'measurements_per_node': {node.node_id: len(node.measurement_history) for node in self.nodes},
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"Simulation completed:")
        logger.info(f"- Total nodes: {self.simulation_results['total_nodes']}")
        logger.info(f"- Total measurements: {self.simulation_results['total_measurements']}")
        logger.info(f"- HyperGraph joins: {self.simulation_results['hypergraph_joins']}")
        logger.info(f"- Cluster breaks: {self.simulation_results['cluster_breaks']}")
        
    async def shutdown(self):
        """Shutdown all nodes"""
        logger.info("Shutting down all nodes...")
        
        disconnect_tasks = []
        for node in self.nodes:
            if node.websocket:
                disconnect_tasks.append(node.disconnect())
        
        if disconnect_tasks:
            await asyncio.gather(*disconnect_tasks, return_exceptions=True)
        
        logger.info("All nodes disconnected")
    
    def print_detailed_results(self):
        """Print detailed simulation results"""
        print("\n" + "="*60)
        print("LAZARUS MESH v5.0 - SIMULATION RESULTS")
        print("="*60)
        
        print(f"\nSimulation Time: {self.simulation_results['timestamp']}")
        print(f"Total Nodes: {self.simulation_results['total_nodes']}")
        print(f"Total Measurements: {self.simulation_results['total_measurements']}")
        print(f"HyperGraph Formations: {self.simulation_results['hypergraph_joins']}")
        print(f"Cluster Fragments: {self.simulation_results['cluster_breaks']}")
        
        print("\nPer-Node Statistics:")
        for node_id, measurement_count in self.simulation_results['measurements_per_node'].items():
            print(f"  Node {node_id}: {measurement_count} measurements")
        
        print("\nDetailed Node Histories:")
        for node in self.nodes:
            print(f"\n--- Node {node.node_id} ---")
            for i, measurement in enumerate(node.measurement_history[-5:]):  # Last 5 measurements
                print(f"  {i+1}. {measurement}")

async def main():
    """Main simulation runner"""
    print("LAZARUS MESH v5.0 - HyperGraph Test Client")
    print("=" * 50)
    print("This client simulates multiple quantum nodes connecting to the HyperGraph network.")
    print("Make sure the bridge_server.py is running before starting this test.")
    print("\nPress Ctrl+C to stop the simulation at any time.")
    
    # Configuration
    num_nodes = 4  # Default to 4 for GHZ-4 state
    simulation_duration = 30  # seconds
    server_uri = "ws://localhost:8765"
    
    simulator = QuantumNetworkSimulator(num_nodes=num_nodes, server_uri=server_uri)
    
    try:
        await simulator.run_simulation(duration=simulation_duration)
        simulator.print_detailed_results()
        
    except KeyboardInterrupt:
        print("\n\nSimulation interrupted by user")
        await simulator.shutdown()
        simulator.print_detailed_results()
        
    except Exception as e:
        print(f"\n\nSimulation error: {e}")
        await simulator.shutdown()
        raise

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nTest client stopped by user")
    except Exception as e:
        print(f"Fatal error: {e}")
        exit(1)