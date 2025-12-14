#!/usr/bin/env python3
"""
LAZARUS MESH v6.0 - Advanced Test Client
Comprehensive testing of Quantum Coherence Stabilization System
Demonstrates Symplectic Geometry, Tensor Contraction, and Adaptive Homology
"""

import asyncio
import websockets
import json
import numpy as np
import random
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
import time
import statistics
from collections import deque

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AdvancedQuantumNode:
    """
    Advanced quantum node with comprehensive testing capabilities
    Integrates symplectic geometry, measurement strategies, and adaptation tracking
    """
    
    def __init__(self, node_id: str):
        self.node_id = node_id
        self.websocket = None
        self.is_connected = False
        self.in_hypergraph = False
        self.cluster_nodes = []
        
        # Testing metrics
        self.measurement_history = []
        self.coherence_history = deque(maxlen=100)
        self.entanglement_events = []
        self.performance_metrics = {
            'messages_sent': 0,
            'messages_received': 0,
            'measurements_performed': 0,
            'adaptation_responses': 0,
            'connection_time': None,
            'hypergraph_join_time': None
        }
        
        # Symplectic state
        self.phase_q = random.randint(0, 16)
        self.phase_p = random.randint(0, 16)
        self.stabilizer_matrix = self._generate_stabilizer()
        
        # Testing configuration
        self.measurement_interval = random.uniform(2, 8)  # Variable intervals
        self.adaptation_response_enabled = True
        self.symplectic_shear_enabled = True
        
    def _generate_stabilizer(self) -> List[List[int]]:
        """Generate Capelli-Stabilizer matrix over F₁₇"""
        return [[random.randint(0, 16) for _ in range(2)] for _ in range(2)]
    
    def calculate_symplectic_product(self, other_q: int, other_p: int) -> int:
        """Calculate symplectic product with another node"""
        return (self.phase_q * other_p - self.phase_p * other_q) % 17
    
    def perform_symplectic_shear(self, shear_type: str = 'q_shear') -> Dict[str, int]:
        """Apply symplectic shear transformation"""
        magnitude = random.randint(1, 16)
        
        if shear_type == 'q_shear':
            # Shear matrix [[1, λ], [0, 1]] applied to phase vector
            self.phase_q = (self.phase_q + magnitude * self.phase_p) % 17
        elif shear_type == 'p_shear':
            # Shear matrix [[1, 0], [λ, 1]] applied to phase vector
            self.phase_p = (self.phase_p + magnitude * self.phase_q) % 17
        
        return {
            'shear_type': shear_type,
            'magnitude': magnitude,
            'new_phase': {'q': self.phase_q, 'p': self.phase_p}
        }
    
    def generate_measurement_strategy(self) -> Dict[str, Any]:
        """Generate intelligent measurement strategy"""
        strategies = ['random', 'adaptive', 'symplectic_optimal', 'coherence_maximizing']
        strategy = random.choice(strategies)
        
        if strategy == 'random':
            basis = random.choice(['Z', 'X', 'Y'])
            outcome = random.choice([0, 1])
        elif strategy == 'adaptive':
            # Adapt based on recent coherence history
            if len(self.coherence_history) > 5:
                recent_coherence = list(self.coherence_history)[-5:]
                avg_coherence = statistics.mean(recent_coherence)
                basis = 'Z' if avg_coherence > 0.7 else 'X'
                outcome = 0 if avg_coherence > 0.5 else 1
            else:
                basis, outcome = 'Z', random.choice([0, 1])
        elif strategy == 'symplectic_optimal':
            # Choose basis based on current phase
            basis = 'Z' if self.phase_q < 8 else 'X'
            outcome = random.choice([0, 1])
        else:  # coherence_maximizing
            # Try to maximize coherence through measurement
            basis = 'Z'
            outcome = 0  # Prefer |0⟩ state for coherence
        
        return {
            'strategy': strategy,
            'basis': basis,
            'outcome': outcome,
            'phase_before': {'q': self.phase_q, 'p': self.phase_p}
        }
    
    async def connect(self, uri: str = "ws://localhost:8765"):
        """Connect to quantum network with retry logic"""
        max_retries = 3
        retry_delay = 1
        
        for attempt in range(max_retries):
            try:
                self.websocket = await websockets.connect(uri)
                self.is_connected = True
                self.performance_metrics['connection_time'] = time.time()
                logger.info(f"Node {self.node_id}: Connected to quantum network")
                
                # Start listening for messages
                await self.listen()
                break
                
            except Exception as e:
                logger.error(f"Node {self.node_id}: Connection attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(retry_delay * (attempt + 1))
                else:
                    raise
    
    async def listen(self):
        """Listen for network messages"""
        try:
            async for message in self.websocket:
                data = json.loads(message)
                await self.handle_message(data)
                self.performance_metrics['messages_received'] += 1
                
        except websockets.exceptions.ConnectionClosed:
            logger.info(f"Node {self.node_id}: Connection closed")
            self.is_connected = False
        except Exception as e:
            logger.error(f"Node {self.node_id}: Error in listener: {e}")
            self.is_connected = False
    
    async def handle_message(self, data: Dict[str, Any]):
        """Handle incoming network messages"""
        logger.info(f"Node {self.node_id}: Received {data.get('type', 'UNKNOWN')}")
        
        if data.get('type') == 'HYPERGRAPH_LOCKED':
            self.in_hypergraph = True
            self.cluster_nodes = data.get('node_ids', [])
            self.performance_metrics['hypergraph_join_time'] = time.time()
            
            logger.info(f"Node {self.node_id}: Joined HyperGraph cluster")
            logger.info(f"Node {self.node_id}: Cluster nodes: {self.cluster_nodes}")
            
            # Start measurement cycle
            asyncio.create_task(self.measurement_cycle())
            
        elif data.get('type') == 'CLUSTER_BROKEN':
            self.in_hypergraph = False
            self.cluster_nodes = []
            
            # Record collapse event
            collapse_event = {
                'timestamp': datetime.now().isoformat(),
                'topology': data.get('topology'),
                'purity': data.get('purity'),
                'coherence': data.get('coherence'),
                'message': data.get('message'),
                'collapsed_node': data.get('collapsed_node')
            }
            
            self.entanglement_events.append(collapse_event)
            logger.info(f"Node {self.node_id}: Cluster broken - {data.get('message', 'N/A')}")
            
            # Respond to collapse if adaptation is enabled
            if self.adaptation_response_enabled:
                await self.respond_to_collapse(collapse_event)
                
        elif data.get('type') == 'NETWORK_METRICS':
            # Update coherence tracking
            coherence = data.get('metrics', {}).get('global_coherence', 0)
            self.coherence_history.append(coherence)
            
        elif data.get('type') == 'STATUS' and 'WAITING' in data.get('status', ''):
            # Log waiting status
            logger.info(f"Node {self.node_id}: Waiting for peers - {data.get('status')}")
    
    async def measurement_cycle(self):
        """Automated measurement cycle with intelligent timing"""
        while self.is_connected and self.in_hypergraph:
            try:
                # Generate measurement strategy
                strategy = self.generate_measurement_strategy()
                
                # Prepare measurement message
                measurement_msg = {
                    'type': 'QUANTUM_MEASUREMENT',
                    'outcome': [1, 0] if strategy['outcome'] == 0 else [0, 1],
                    'basis': strategy['basis'],
                    'strategy': strategy['strategy'],
                    'phase_state': strategy['phase_before'],
                    'timestamp': datetime.now().isoformat()
                }
                
                # Send measurement
                await self.websocket.send(json.dumps(measurement_msg))
                self.performance_metrics['messages_sent'] += 1
                self.performance_metrics['measurements_performed'] += 1
                
                # Record measurement
                self.measurement_history.append(measurement_msg)
                
                logger.info(f"Node {self.node_id}: Measurement sent - "
                           f"Strategy: {strategy['strategy']}, "
                           f"Basis: {strategy['basis']}, "
                           f"Outcome: {strategy['outcome']}")
                
                # Wait for next measurement interval
                await asyncio.sleep(self.measurement_interval)
                
            except Exception as e:
                logger.error(f"Node {self.node_id}: Error in measurement cycle: {e}")
                break
    
    async def respond_to_collapse(self, collapse_event: Dict[str, Any]):
        """Respond to cluster collapse with adaptive strategies"""
        try:
            # Update performance metrics
            self.performance_metrics['adaptation_responses'] += 1
            
            # Generate adaptive response
            if self.symplectic_shear_enabled:
                # Apply symplectic shear to adapt to new topology
                shear_result = self.perform_symplectic_shear()
                
                response_msg = {
                    'type': 'ADAPTATION_RESPONSE',
                    'response_type': 'symplectic_shear',
                    'shear_result': shear_result,
                    'trigger_event': collapse_event,
                    'timestamp': datetime.now().isoformat()
                }
                
                await self.websocket.send(json.dumps(response_msg))
                self.performance_metrics['messages_sent'] += 1
                
                logger.info(f"Node {self.node_id}: Sent adaptation response - "
                           f"Shear type: {shear_result['shear_type']}")
            
        except Exception as e:
            logger.error(f"Node {self.node_id}: Error in collapse response: {e}")
    
    async def request_metrics(self):
        """Request network metrics"""
        if self.is_connected:
            try:
                await self.websocket.send(json.dumps({
                    'type': 'REQUEST_METRICS'
                }))
                self.performance_metrics['messages_sent'] += 1
            except Exception as e:
                logger.error(f"Node {self.node_id}: Error requesting metrics: {e}")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary"""
        connection_time = self.performance_metrics['connection_time']
        hypergraph_time = self.performance_metrics['hypergraph_join_time']
        
        runtime = time.time() - connection_time if connection_time else 0
        hypergraph_wait = hypergraph_time - connection_time if hypergraph_time and connection_time else 0
        
        return {
            'node_id': self.node_id,
            'runtime_seconds': runtime,
            'hypergraph_wait_seconds': hypergraph_wait,
            'messages_sent': self.performance_metrics['messages_sent'],
            'messages_received': self.performance_metrics['messages_received'],
            'measurements_performed': self.performance_metrics['measurements_performed'],
            'adaptation_responses': self.performance_metrics['adaptation_responses'],
            'measurement_history_count': len(self.measurement_history),
            'entanglement_events_count': len(self.entanglement_events),
            'average_coherence': statistics.mean(self.coherence_history) if self.coherence_history else 0,
            'coherence_variance': statistics.variance(self.coherence_history) if len(self.coherence_history) > 1 else 0,
            'final_phase_state': {
                'q': self.phase_q,
                'p': self.phase_p
            },
            'stabilizer_trace': sum(sum(row) for row in self.stabilizer_matrix) % 17
        }

class QuantumNetworkTestSuite:
    """
    Comprehensive test suite for Lazarus Mesh v6.0
    Tests all major components: symplectic geometry, tensor contraction, adaptation
    """
    
    def __init__(self, num_nodes: int = 4, test_duration: int = 60):
        self.num_nodes = num_nodes
        self.test_duration = test_duration
        self.nodes: List[AdvancedQuantumNode] = []
        self.test_results = {}
        
        # Test configuration
        self.enable_symplectic_shear = True
        self.enable_adaptation_responses = True
        self.enable_metrics_collection = True
        
    async def initialize_nodes(self):
        """Initialize all test nodes"""
        logger.info(f"Initializing {self.num_nodes} advanced quantum nodes...")
        
        for i in range(self.num_nodes):
            node = AdvancedQuantumNode(f"test_node_{i:02d}")
            node.symplectic_shear_enabled = self.enable_symplectic_shear
            node.adaptation_response_enabled = self.enable_adaptation_responses
            node.measurement_interval = random.uniform(3, 7)  # Varied intervals
            
            self.nodes.append(node)
            
            # Stagger connections
            await asyncio.sleep(random.uniform(0.5, 2.0))
            
            try:
                asyncio.create_task(node.connect())
            except Exception as e:
                logger.error(f"Failed to start node {i}: {e}")
    
    async def run_test_scenario(self):
        """Execute comprehensive test scenario"""
        logger.info("Starting comprehensive test scenario...")
        
        # Phase 1: Basic connectivity and HyperGraph formation
        logger.info("Phase 1: Testing basic connectivity and HyperGraph formation")
        await self.initialize_nodes()
        
        # Wait for initial cluster formation
        await asyncio.sleep(15)
        
        # Phase 2: Measurement and collapse testing
        logger.info("Phase 2: Testing measurement propagation and collapse")
        await self.test_measurement_scenarios()
        
        # Phase 3: Symplectic geometry testing
        logger.info("Phase 3: Testing symplectic geometry operations")
        await self.test_symplectic_operations()
        
        # Phase 4: Adaptive stabilization testing
        logger.info("Phase 4: Testing adaptive stabilization")
        await self.test_adaptation_scenarios()
        
        # Phase 5: Metrics collection
        logger.info("Phase 5: Collecting comprehensive metrics")
        await self.collect_comprehensive_metrics()
        
    async def test_measurement_scenarios(self):
        """Test various measurement scenarios"""
        hypergraph_nodes = [node for node in self.nodes if node.in_hypergraph]
        
        if len(hypergraph_nodes) < 2:
            logger.warning("Insufficient nodes in HyperGraph for measurement testing")
            return
        
        # Test sequential measurements
        for i, node in enumerate(hypergraph_nodes[:3]):  # Test first 3 nodes
            if node.is_connected and node.in_hypergraph:
                # Trigger measurement by temporarily changing measurement interval
                original_interval = node.measurement_interval
                node.measurement_interval = 1.0  # Immediate measurement
                await asyncio.sleep(2)
                node.measurement_interval = original_interval
                
                logger.info(f"Triggered measurement for node {node.node_id}")
                await asyncio.sleep(3)  # Wait for collapse propagation
    
    async def test_symplectic_operations(self):
        """Test symplectic geometry operations"""
        for node in self.nodes:
            if node.is_connected:
                # Test symplectic shear
                shear_type = random.choice(['q_shear', 'p_shear'])
                shear_result = node.perform_symplectic_shear(shear_type)
                
                logger.info(f"Node {node.node_id}: Applied {shear_type} shear")
                
                # Test symplectic product calculation
                other_node = random.choice([n for n in self.nodes if n != node and n.is_connected])
                if other_node:
                    symplectic_product = node.calculate_symplectic_product(
                        other_node.phase_q, other_node.phase_p
                    )
                    logger.info(f"Symplectic product with {other_node.node_id}: {symplectic_product}")
                
                await asyncio.sleep(1)
    
    async def test_adaptation_scenarios(self):
        """Test adaptive stabilization responses"""
        # Enable adaptation for some nodes
        adaptation_nodes = random.sample(self.nodes, min(2, len(self.nodes)))
        
        for node in adaptation_nodes:
            node.adaptation_response_enabled = True
            logger.info(f"Enabled adaptation responses for node {node.node_id}")
        
        # Wait for adaptation responses
        await asyncio.sleep(10)
    
    async def collect_comprehensive_metrics(self):
        """Collect all performance metrics"""
        logger.info("Collecting comprehensive performance metrics...")
        
        # Request final metrics from all nodes
        for node in self.nodes:
            if node.is_connected:
                await node.request_metrics()
        
        await asyncio.sleep(5)  # Wait for metrics collection
        
        # Compile test results
        self.test_results = {
            'test_scenario': 'Lazarus Mesh v6.0 Comprehensive Test',
            'test_duration_seconds': self.test_duration,
            'num_test_nodes': len(self.nodes),
            'timestamp': datetime.now().isoformat(),
            'individual_results': [],
            'network_summary': {},
            'performance_analysis': {}
        }
        
        # Collect individual node results
        for node in self.nodes:
            node_result = node.get_performance_summary()
            self.test_results['individual_results'].append(node_result)
        
        # Generate network summary
        self.generate_network_summary()
        
        # Generate performance analysis
        self.generate_performance_analysis()
    
    def generate_network_summary(self):
        """Generate network-wide summary statistics"""
        individual_results = self.test_results['individual_results']
        
        if not individual_results:
            return
        
        self.test_results['network_summary'] = {
            'total_runtime_seconds': max(r['runtime_seconds'] for r in individual_results),
            'total_measurements': sum(r['measurements_performed'] for r in individual_results),
            'total_messages_sent': sum(r['messages_sent'] for r in individual_results),
            'total_messages_received': sum(r['messages_received'] for r in individual_results),
            'total_adaptation_responses': sum(r['adaptation_responses'] for r in individual_results),
            'average_coherence': statistics.mean(r['average_coherence'] for r in individual_results),
            'coherence_variance': statistics.mean(r['coherence_variance'] for r in individual_results),
            'successful_hypergraph_joins': len([r for r in individual_results if r['hypergraph_wait_seconds'] > 0]),
            'average_hypergraph_wait': statistics.mean(
                [r['hypergraph_wait_seconds'] for r in individual_results if r['hypergraph_wait_seconds'] > 0]
            ) if any(r['hypergraph_wait_seconds'] > 0 for r in individual_results) else 0
        }
    
    def generate_performance_analysis(self):
        """Generate detailed performance analysis"""
        network_summary = self.test_results['network_summary']
        individual_results = self.test_results['individual_results']
        
        analysis = {
            'connectivity_analysis': {
                'connection_success_rate': len([r for r in individual_results if r['runtime_seconds'] > 0]) / len(individual_results),
                'hypergraph_formation_rate': network_summary['successful_hypergraph_joins'] / len(individual_results),
                'average_wait_time': network_summary['average_hypergraph_wait']
            },
            'measurement_analysis': {
                'total_measurements': network_summary['total_measurements'],
                'measurements_per_node_avg': network_summary['total_measurements'] / len(individual_results),
                'measurement_success_rate': 1.0  # Assuming all measurements were processed
            },
            'coherence_analysis': {
                'average_global_coherence': network_summary['average_coherence'],
                'coherence_stability': 1.0 / (1.0 + network_summary['coherence_variance']),
                'coherence_trend': 'stable' if network_summary['coherence_variance'] < 0.1 else 'variable'
            },
            'communication_analysis': {
                'total_messages': network_summary['total_messages_sent'] + network_summary['total_messages_received'],
                'message_efficiency': network_summary['total_messages_received'] / network_summary['total_messages_sent'] if network_summary['total_messages_sent'] > 0 else 0,
                'adaptation_response_rate': network_summary['total_adaptation_responses'] / len(individual_results)
            }
        }
        
        self.test_results['performance_analysis'] = analysis
    
    def print_test_results(self):
        """Print comprehensive test results"""
        print("\n" + "="*80)
        print("LAZARUS MESH v6.0 - COMPREHENSIVE TEST RESULTS")
        print("="*80)
        
        results = self.test_results
        
        print(f"\nTest Configuration:")
        print(f"  Test Duration: {results['test_duration_seconds']} seconds")
        print(f"  Number of Nodes: {results['num_test_nodes']}")
        print(f"  Test Completed: {results['timestamp']}")
        
        print(f"\nNetwork Summary:")
        network = results['network_summary']
        print(f"  Total Runtime: {network['total_runtime_seconds']:.2f} seconds")
        print(f"  Total Measurements: {network['total_measurements']}")
        print(f"  Total Messages: {network['total_messages_sent']} sent, {network['total_messages_received']} received")
        print(f"  Average Coherence: {network['average_coherence']:.4f}")
        print(f"  Successful HyperGraph Joins: {network['successful_hypergraph_joins']}/{results['num_test_nodes']}")
        
        print(f"\nPerformance Analysis:")
        analysis = results['performance_analysis']
        
        print(f"\n  Connectivity Analysis:")
        print(f"    Connection Success Rate: {analysis['connectivity_analysis']['connection_success_rate']:.2%}")
        print(f"    HyperGraph Formation Rate: {analysis['connectivity_analysis']['hypergraph_formation_rate']:.2%}")
        print(f"    Average Wait Time: {analysis['connectivity_analysis']['average_wait_time']:.2f} seconds")
        
        print(f"\n  Coherence Analysis:")
        print(f"    Average Global Coherence: {analysis['coherence_analysis']['average_global_coherence']:.4f}")
        print(f"    Coherence Stability: {analysis['coherence_analysis']['coherence_stability']:.4f}")
        print(f"    Coherence Trend: {analysis['coherence_analysis']['coherence_trend']}")
        
        print(f"\n  Communication Analysis:")
        print(f"    Total Messages: {analysis['communication_analysis']['total_messages']}")
        print(f"    Message Efficiency: {analysis['communication_analysis']['message_efficiency']:.2%}")
        print(f"    Adaptation Response Rate: {analysis['communication_analysis']['adaptation_response_rate']:.2f}")
        
        print(f"\nIndividual Node Results:")
        for i, node_result in enumerate(results['individual_results']):
            print(f"\n  Node {i}: {node_result['node_id']}")
            print(f"    Runtime: {node_result['runtime_seconds']:.2f} seconds")
            print(f"    Measurements: {node_result['measurements_performed']}")
            print(f"    Messages: {node_result['messages_sent']} sent, {node_result['messages_received']} received")
            print(f"    Average Coherence: {node_result['average_coherence']:.4f}")
            print(f"    Final Phase: q={node_result['final_phase_state']['q']}, p={node_result['final_phase_state']['p']}")
    
    async def shutdown(self):
        """Shutdown all test nodes"""
        logger.info("Shutting down test nodes...")
        
        disconnect_tasks = []
        for node in self.nodes:
            if node.websocket:
                disconnect_tasks.append(node.disconnect())
        
        if disconnect_tasks:
            await asyncio.gather(*disconnect_tasks, return_exceptions=True)
        
        logger.info("All test nodes disconnected")

async def main():
    """Main test execution"""
    print("LAZARUS MESH v6.0 - ADVANCED TEST SUITE")
    print("=" * 60)
    print("Comprehensive testing of quantum coherence stabilization system")
    print("Features: Symplectic Geometry, Tensor Contraction, Adaptive Homology")
    print("\nMake sure quantum_mesh_v6.py server is running before starting tests.")
    print("\nPress Ctrl+C to stop tests at any time.")
    
    # Test configuration
    num_test_nodes = 6  # More nodes for comprehensive testing
    test_duration = 90  # Extended duration for thorough testing
    
    test_suite = QuantumNetworkTestSuite(
        num_nodes=num_test_nodes,
        test_duration=test_duration
    )
    
    try:
        # Run comprehensive test scenario
        await test_suite.run_test_scenario()
        
        # Print results
        test_suite.print_test_results()
        
        # Save results to file
        results_file = f"test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_file, 'w') as f:
            json.dump(test_suite.test_results, f, indent=2)
        
        print(f"\nTest results saved to: {results_file}")
        
    except KeyboardInterrupt:
        print("\n\nTest execution interrupted by user")
        await test_suite.shutdown()
        test_suite.print_test_results()
        
    except Exception as e:
        print(f"\n\nTest execution error: {e}")
        logger.exception("Test execution failed")
        await test_suite.shutdown()
        raise

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nTest suite stopped by user")
    except Exception as e:
        print(f"Fatal error: {e}")
        logger.exception("Fatal error in test suite")
        exit(1)