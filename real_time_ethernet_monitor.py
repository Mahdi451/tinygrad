#!/usr/bin/env python3
"""
Real-Time Ethernet Testing Monitor
=================================

Live anomaly detection system for Ethernet testing that monitors PAM4/NRZ signals,
CTLE performance, ISI, and BER in real-time during data collection.

This system is designed to integrate with test equipment for immediate anomaly
detection and alerting during active Ethernet validation testing.

Key Features:
- Real-time signal analysis as data is collected
- Immediate alerts for spec violations and anomalies
- Live dashboards for test operators
- Historical trend tracking
- Integration hooks for test equipment
"""

import time
import threading
import queue
import logging
from typing import Dict, List, Callable, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from tinygrad import Tensor

# Optional imports
try:
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    from matplotlib.patches import Rectangle
    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False

class AlertSeverity(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"

@dataclass
class RealTimeAlert:
    """Real-time alert structure"""
    timestamp: float
    severity: AlertSeverity
    parameter: str
    message: str
    value: float
    threshold: float
    recommended_action: str = ""

@dataclass
class EthernetTestConfig:
    """Real-time testing configuration"""
    # Signal parameters
    signal_type: str = "PAM4"  # "PAM4" or "NRZ"
    sample_rate: float = 25e9  # 25 GSa/s
    symbol_rate: float = 10e9  # 10 GBaud
    
    # Real-time processing
    buffer_size: int = 1000  # Samples per analysis window
    analysis_interval: float = 0.1  # Analysis every 100ms
    alert_cooldown: float = 5.0  # Minimum time between duplicate alerts
    
    # Thresholds
    pam4_levels: List[float] = field(default_factory=lambda: [-0.75, -0.25, 0.25, 0.75])
    eye_height_min: float = 0.4
    eye_width_min: float = 0.6
    ber_threshold: float = 1e-12
    isi_threshold: float = 0.15
    ctle_gain_min: float = 0.0
    ctle_gain_max: float = 20.0
    
    # Alert configuration
    enable_audio_alerts: bool = True
    enable_email_alerts: bool = False
    operator_notification: bool = True

class RealTimeEthernetMonitor:
    """
    Real-time Ethernet testing monitor with live anomaly detection
    """
    
    def __init__(self, config: EthernetTestConfig):
        """
        Initialize the real-time monitor
        
        Args:
            config: Testing configuration parameters
        """
        self.config = config
        self.running = False
        
        # Data queues for real-time processing
        self.signal_queue = queue.Queue(maxsize=10000)
        self.alert_queue = queue.Queue()
        
        # Processing threads
        self.analysis_thread = None
        self.alert_thread = None
        self.plotting_thread = None
        
        # Historical data for trending
        self.history = {
            'timestamps': [],
            'eye_height': [],
            'eye_width': [],
            'ber': [],
            'isi_ratio': [],
            'ctle_gain': [],
            'alerts': []
        }
        
        # Alert management
        self.last_alert_time = {}
        self.total_alerts = 0
        
        # Performance metrics
        self.samples_processed = 0
        self.analysis_time_history = []
        
        # Setup logging
        logging.basicConfig(level=logging.INFO,
                          format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)
        
        # Live plotting setup
        if HAS_PLOTTING:
            self.setup_live_plotting()
    
    def setup_live_plotting(self):
        """Setup live plotting for real-time monitoring"""
        plt.ion()
        self.fig, self.axes = plt.subplots(2, 2, figsize=(15, 10))
        self.fig.suptitle('Real-Time Ethernet Testing Monitor', fontsize=16)
        
        # Configure subplots
        self.axes[0, 0].set_title('Signal Level Monitoring')
        self.axes[0, 0].set_ylabel('Amplitude (V)')
        self.axes[0, 0].grid(True, alpha=0.3)
        
        self.axes[0, 1].set_title('Eye Diagram Metrics')
        self.axes[0, 1].set_ylabel('Eye Height/Width')
        self.axes[0, 1].grid(True, alpha=0.3)
        
        self.axes[1, 0].set_title('BER Trending')
        self.axes[1, 0].set_ylabel('Bit Error Rate')
        self.axes[1, 0].set_yscale('log')
        self.axes[1, 0].grid(True, alpha=0.3)
        
        self.axes[1, 1].set_title('ISI and CTLE Monitoring')
        self.axes[1, 1].set_ylabel('ISI Ratio / CTLE Gain')
        self.axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
    
    def add_data_point(self, timestamp: float, signal_data: np.ndarray, 
                      metadata: Optional[Dict] = None):
        """
        Add new data point for real-time analysis
        
        Args:
            timestamp: Data timestamp
            signal_data: Signal samples
            metadata: Additional data (BER info, CTLE settings, etc.)
        """
        try:
            data_point = {
                'timestamp': timestamp,
                'signal': signal_data,
                'metadata': metadata or {}
            }
            
            if not self.signal_queue.full():
                self.signal_queue.put(data_point, block=False)
                self.samples_processed += len(signal_data)
            else:
                self.logger.warning("Signal queue full, dropping data point")
                
        except queue.Full:
            self.logger.warning("Failed to add data point - queue full")
    
    def analyze_signal_window(self, signal_data: np.ndarray, 
                            metadata: Dict) -> Dict:
        """
        Analyze a window of signal data for anomalies
        
        Args:
            signal_data: Signal samples to analyze
            metadata: Associated metadata
            
        Returns:
            Analysis results with detected anomalies
        """
        start_time = time.time()
        
        # Convert to tensor for analysis
        signal_tensor = Tensor(signal_data.astype(np.float32))
        
        results = {
            'timestamp': time.time(),
            'anomalies': [],
            'metrics': {}
        }
        
        # 1. Signal level analysis
        if self.config.signal_type == "PAM4":
            level_anomalies = self._analyze_pam4_levels(signal_tensor)
            results['anomalies'].extend(level_anomalies)
        else:
            level_anomalies = self._analyze_nrz_levels(signal_tensor)
            results['anomalies'].extend(level_anomalies)
        
        # 2. Eye diagram analysis
        eye_metrics = self._analyze_eye_diagram_realtime(signal_tensor)
        results['metrics']['eye'] = eye_metrics
        
        # Check eye diagram thresholds
        if eye_metrics['height'] < self.config.eye_height_min:
            results['anomalies'].append(RealTimeAlert(
                timestamp=time.time(),
                severity=AlertSeverity.CRITICAL,
                parameter="eye_height",
                message=f"Eye height {eye_metrics['height']:.3f}V below minimum {self.config.eye_height_min:.3f}V",
                value=eye_metrics['height'],
                threshold=self.config.eye_height_min,
                recommended_action="Check signal integrity, verify test setup"
            ))
        
        if eye_metrics['width'] < self.config.eye_width_min:
            results['anomalies'].append(RealTimeAlert(
                timestamp=time.time(),
                severity=AlertSeverity.CRITICAL,
                parameter="eye_width",
                message=f"Eye width {eye_metrics['width']:.3f}UI below minimum {self.config.eye_width_min:.3f}UI",
                value=eye_metrics['width'],
                threshold=self.config.eye_width_min,
                recommended_action="Check jitter, verify clock recovery"
            ))
        
        # 3. ISI analysis
        isi_metrics = self._analyze_isi_realtime(signal_tensor)
        results['metrics']['isi'] = isi_metrics
        
        if isi_metrics['isi_ratio'] > self.config.isi_threshold:
            results['anomalies'].append(RealTimeAlert(
                timestamp=time.time(),
                severity=AlertSeverity.WARNING,
                parameter="isi_ratio",
                message=f"ISI ratio {isi_metrics['isi_ratio']:.3f} exceeds threshold {self.config.isi_threshold:.3f}",
                value=isi_metrics['isi_ratio'],
                threshold=self.config.isi_threshold,
                recommended_action="Check channel response, adjust equalization"
            ))
        
        # 4. BER analysis (if data available)
        if 'tx_bits' in metadata and 'rx_bits' in metadata:
            ber_metrics = self._analyze_ber_realtime(metadata['tx_bits'], metadata['rx_bits'])
            results['metrics']['ber'] = ber_metrics
            
            if ber_metrics['ber'] > self.config.ber_threshold:
                results['anomalies'].append(RealTimeAlert(
                    timestamp=time.time(),
                    severity=AlertSeverity.CRITICAL,
                    parameter="ber",
                    message=f"BER {ber_metrics['ber']:.2e} exceeds threshold {self.config.ber_threshold:.2e}",
                    value=ber_metrics['ber'],
                    threshold=self.config.ber_threshold,
                    recommended_action="Check link quality, verify signal levels"
                ))
        
        # 5. CTLE analysis (if data available)
        if 'ctle_gain' in metadata:
            ctle_gain = metadata['ctle_gain']
            results['metrics']['ctle_gain'] = ctle_gain
            
            if ctle_gain < self.config.ctle_gain_min or ctle_gain > self.config.ctle_gain_max:
                results['anomalies'].append(RealTimeAlert(
                    timestamp=time.time(),
                    severity=AlertSeverity.WARNING,
                    parameter="ctle_gain",
                    message=f"CTLE gain {ctle_gain:.1f}dB outside range [{self.config.ctle_gain_min:.1f}, {self.config.ctle_gain_max:.1f}]",
                    value=ctle_gain,
                    threshold=self.config.ctle_gain_max,
                    recommended_action="Check CTLE adaptation, verify channel response"
                ))
        
        # Record analysis time for performance monitoring
        analysis_time = time.time() - start_time
        self.analysis_time_history.append(analysis_time)
        if len(self.analysis_time_history) > 100:
            self.analysis_time_history.pop(0)
        
        return results
    
    def _analyze_pam4_levels(self, signal: Tensor) -> List[RealTimeAlert]:
        """Analyze PAM4 signal levels for anomalies"""
        alerts = []
        
        # Classify signal levels
        signal_np = signal.numpy()
        level_counts = np.zeros(4)
        
        for sample in signal_np:
            # Find closest PAM4 level
            distances = [abs(sample - level) for level in self.config.pam4_levels]
            closest_level = np.argmin(distances)
            level_counts[closest_level] += 1
            
            # Check if sample is too far from any level
            min_distance = min(distances)
            if min_distance > 0.1:  # Threshold for level violation
                alerts.append(RealTimeAlert(
                    timestamp=time.time(),
                    severity=AlertSeverity.WARNING,
                    parameter="signal_level",
                    message=f"Signal sample {sample:.3f}V far from any PAM4 level",
                    value=sample,
                    threshold=0.1,
                    recommended_action="Check signal integrity, noise levels"
                ))
        
        # Check level distribution balance
        total_samples = len(signal_np)
        expected_per_level = total_samples / 4
        for i, count in enumerate(level_counts):
            if abs(count - expected_per_level) > expected_per_level * 0.5:  # 50% imbalance
                alerts.append(RealTimeAlert(
                    timestamp=time.time(),
                    severity=AlertSeverity.INFO,
                    parameter="level_distribution",
                    message=f"PAM4 level {i} imbalanced: {count} samples vs expected {expected_per_level:.0f}",
                    value=count,
                    threshold=expected_per_level * 1.5,
                    recommended_action="Check data pattern, verify test sequence"
                ))
        
        return alerts
    
    def _analyze_nrz_levels(self, signal: Tensor) -> List[RealTimeAlert]:
        """Analyze NRZ signal levels for anomalies"""
        alerts = []
        
        # Simple threshold-based NRZ analysis
        signal_np = signal.numpy()
        threshold = 0.5  # Midpoint between 0 and 1
        
        # Check for samples in transition region (too long)
        transition_samples = np.sum((signal_np > 0.3) & (signal_np < 0.7))
        if transition_samples > len(signal_np) * 0.1:  # More than 10% in transition
            alerts.append(RealTimeAlert(
                timestamp=time.time(),
                severity=AlertSeverity.WARNING,
                parameter="nrz_transitions",
                message=f"Excessive samples in transition region: {transition_samples}",
                value=transition_samples,
                threshold=len(signal_np) * 0.1,
                recommended_action="Check rise/fall times, signal bandwidth"
            ))
        
        return alerts
    
    def _analyze_eye_diagram_realtime(self, signal: Tensor) -> Dict:
        """Fast eye diagram analysis for real-time monitoring"""
        signal_np = signal.numpy()
        samples_per_symbol = int(self.config.sample_rate / self.config.symbol_rate)
        
        if len(signal_np) < samples_per_symbol * 3:
            # Not enough data for eye analysis
            return {'height': 0.0, 'width': 0.0, 'opening': 0.0}
        
        # Reshape into symbol periods
        num_symbols = len(signal_np) // samples_per_symbol
        eye_data = signal_np[:num_symbols * samples_per_symbol].reshape(num_symbols, samples_per_symbol)
        
        # Calculate eye metrics
        center_idx = samples_per_symbol // 2
        center_samples = eye_data[:, center_idx]
        
        eye_height = float(np.max(center_samples) - np.min(center_samples))
        
        # Simplified eye width calculation
        eye_width = 1.0 - (np.std(center_samples) / np.mean(np.abs(center_samples)))
        eye_width = max(0.0, min(1.0, eye_width))  # Clamp to [0, 1]
        
        # Eye opening ratio
        signal_range = np.max(signal_np) - np.min(signal_np)
        eye_opening = eye_height / signal_range if signal_range > 0 else 0.0
        
        return {
            'height': eye_height,
            'width': eye_width,
            'opening': eye_opening
        }
    
    def _analyze_isi_realtime(self, signal: Tensor) -> Dict:
        """Fast ISI analysis for real-time monitoring"""
        signal_np = signal.numpy()
        
        # Simplified ISI calculation based on signal variation
        # More sophisticated analysis would require symbol classification
        
        # Calculate local variations
        if len(signal_np) > 10:
            variations = np.diff(signal_np)
            isi_estimate = np.std(variations) / np.mean(np.abs(signal_np))
        else:
            isi_estimate = 0.0
        
        return {
            'isi_ratio': float(isi_estimate),
            'timing_margin': max(0.0, 1.0 - isi_estimate * 2)
        }
    
    def _analyze_ber_realtime(self, tx_bits: List[int], rx_bits: List[int]) -> Dict:
        """Fast BER analysis for real-time monitoring"""
        if len(tx_bits) != len(rx_bits) or len(tx_bits) == 0:
            return {'ber': 0.0, 'error_count': 0, 'total_bits': 0}
        
        # Convert to tensors for efficient comparison
        tx_tensor = Tensor(tx_bits)
        rx_tensor = Tensor(rx_bits)
        
        errors = (tx_tensor != rx_tensor).float()
        error_count = errors.sum().item()
        total_bits = len(tx_bits)
        ber = error_count / total_bits
        
        return {
            'ber': float(ber),
            'error_count': int(error_count),
            'total_bits': total_bits
        }
    
    def _analysis_worker(self):
        """Worker thread for continuous signal analysis"""
        self.logger.info("Analysis worker started")
        
        buffer = []
        
        while self.running:
            try:
                # Collect data points for analysis window
                while len(buffer) < self.config.buffer_size:
                    try:
                        data_point = self.signal_queue.get(timeout=0.01)
                        buffer.append(data_point)
                    except queue.Empty:
                        if not self.running:
                            break
                        continue
                
                if not buffer:
                    continue
                
                # Combine buffer data for analysis
                timestamps = [dp['timestamp'] for dp in buffer]
                signals = [dp['signal'] for dp in buffer]
                combined_signal = np.concatenate(signals)
                
                # Use metadata from most recent data point
                metadata = buffer[-1]['metadata']
                
                # Perform analysis
                analysis_results = self.analyze_signal_window(combined_signal, metadata)
                
                # Store metrics in history
                current_time = time.time()
                self.history['timestamps'].append(current_time)
                
                if 'eye' in analysis_results['metrics']:
                    self.history['eye_height'].append(analysis_results['metrics']['eye']['height'])
                    self.history['eye_width'].append(analysis_results['metrics']['eye']['width'])
                else:
                    self.history['eye_height'].append(0.0)
                    self.history['eye_width'].append(0.0)
                
                if 'ber' in analysis_results['metrics']:
                    self.history['ber'].append(analysis_results['metrics']['ber']['ber'])
                else:
                    self.history['ber'].append(0.0)
                
                if 'isi' in analysis_results['metrics']:
                    self.history['isi_ratio'].append(analysis_results['metrics']['isi']['isi_ratio'])
                else:
                    self.history['isi_ratio'].append(0.0)
                
                if 'ctle_gain' in analysis_results['metrics']:
                    self.history['ctle_gain'].append(analysis_results['metrics']['ctle_gain'])
                else:
                    self.history['ctle_gain'].append(0.0)
                
                # Keep history limited
                max_history = 1000
                for key in self.history:
                    if len(self.history[key]) > max_history:
                        self.history[key] = self.history[key][-max_history:]
                
                # Queue alerts
                for alert in analysis_results['anomalies']:
                    self._queue_alert(alert)
                
                # Clear buffer
                buffer.clear()
                
                # Wait for next analysis interval
                time.sleep(self.config.analysis_interval)
                
            except Exception as e:
                self.logger.error(f"Analysis worker error: {e}")
                time.sleep(0.1)
        
        self.logger.info("Analysis worker stopped")
    
    def _queue_alert(self, alert: RealTimeAlert):
        """Queue an alert with duplicate suppression"""
        current_time = time.time()
        alert_key = f"{alert.parameter}_{alert.severity.value}"
        
        # Check for cooldown period
        if alert_key in self.last_alert_time:
            if current_time - self.last_alert_time[alert_key] < self.config.alert_cooldown:
                return  # Skip duplicate alert
        
        # Queue the alert
        try:
            self.alert_queue.put(alert, block=False)
            self.last_alert_time[alert_key] = current_time
            self.total_alerts += 1
        except queue.Full:
            self.logger.warning("Alert queue full, dropping alert")
    
    def _alert_worker(self):
        """Worker thread for processing alerts"""
        self.logger.info("Alert worker started")
        
        while self.running:
            try:
                alert = self.alert_queue.get(timeout=1.0)
                
                # Process the alert
                self._process_alert(alert)
                
                # Store in history
                self.history['alerts'].append({
                    'timestamp': alert.timestamp,
                    'severity': alert.severity.value,
                    'parameter': alert.parameter,
                    'message': alert.message,
                    'value': alert.value,
                    'threshold': alert.threshold
                })
                
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"Alert worker error: {e}")
        
        self.logger.info("Alert worker stopped")
    
    def _process_alert(self, alert: RealTimeAlert):
        """Process and display an alert"""
        # Log the alert
        if alert.severity == AlertSeverity.CRITICAL:
            self.logger.critical(f"CRITICAL: {alert.message}")
        elif alert.severity == AlertSeverity.WARNING:
            self.logger.warning(f"WARNING: {alert.message}")
        else:
            self.logger.info(f"INFO: {alert.message}")
        
        # Console output with color
        severity_colors = {
            AlertSeverity.CRITICAL: '\\033[91m',  # Red
            AlertSeverity.WARNING: '\\033[93m',   # Yellow
            AlertSeverity.INFO: '\\033[94m'       # Blue
        }
        reset_color = '\\033[0m'
        
        color = severity_colors.get(alert.severity, '')
        print(f"{color}[{alert.severity.value.upper()}] {alert.timestamp:.2f}: {alert.message}{reset_color}")
        
        if alert.recommended_action:
            print(f"   → Recommended action: {alert.recommended_action}")
        
        # Audio alert for critical issues
        if (alert.severity == AlertSeverity.CRITICAL and 
            self.config.enable_audio_alerts):
            try:
                # Simple beep (works on most systems)
                print('\\a')  # ASCII bell character
            except:
                pass
        
        # Email alerts (if configured)
        if (alert.severity == AlertSeverity.CRITICAL and 
            self.config.enable_email_alerts):
            self._send_email_alert(alert)
    
    def _send_email_alert(self, alert: RealTimeAlert):
        """Send email alert (placeholder implementation)"""
        # This would integrate with your email system
        self.logger.info(f"Email alert would be sent: {alert.message}")
    
    def _update_live_plot(self):
        """Update live plotting display"""
        if not HAS_PLOTTING or len(self.history['timestamps']) < 2:
            return
        
        try:
            # Clear previous plots
            for ax in self.axes.flatten():
                ax.clear()
            
            times = self.history['timestamps'][-100:]  # Last 100 points
            
            # Plot 1: Signal quality metrics
            if self.history['eye_height']:
                eye_heights = self.history['eye_height'][-100:]
                self.axes[0, 0].plot(times, eye_heights, 'b-', label='Eye Height')
                self.axes[0, 0].axhline(y=self.config.eye_height_min, color='r', linestyle='--', label='Min Threshold')
                self.axes[0, 0].set_title('Eye Height Monitoring')
                self.axes[0, 0].set_ylabel('Eye Height (V)')
                self.axes[0, 0].legend()
                self.axes[0, 0].grid(True, alpha=0.3)
            
            # Plot 2: Eye width
            if self.history['eye_width']:
                eye_widths = self.history['eye_width'][-100:]
                self.axes[0, 1].plot(times, eye_widths, 'g-', label='Eye Width')
                self.axes[0, 1].axhline(y=self.config.eye_width_min, color='r', linestyle='--', label='Min Threshold')
                self.axes[0, 1].set_title('Eye Width Monitoring')
                self.axes[0, 1].set_ylabel('Eye Width (UI)')
                self.axes[0, 1].legend()
                self.axes[0, 1].grid(True, alpha=0.3)
            
            # Plot 3: BER
            if self.history['ber'] and any(ber > 0 for ber in self.history['ber'][-100:]):
                bers = [max(ber, 1e-15) for ber in self.history['ber'][-100:]]  # Avoid log(0)
                self.axes[1, 0].semilogy(times, bers, 'r-', label='BER')
                self.axes[1, 0].axhline(y=self.config.ber_threshold, color='r', linestyle='--', label='Threshold')
                self.axes[1, 0].set_title('BER Monitoring')
                self.axes[1, 0].set_ylabel('Bit Error Rate')
                self.axes[1, 0].legend()
                self.axes[1, 0].grid(True, alpha=0.3)
            
            # Plot 4: ISI and CTLE
            if self.history['isi_ratio']:
                isi_ratios = self.history['isi_ratio'][-100:]
                self.axes[1, 1].plot(times, isi_ratios, 'm-', label='ISI Ratio')
                self.axes[1, 1].axhline(y=self.config.isi_threshold, color='r', linestyle='--', label='ISI Threshold')
                
                if self.history['ctle_gain']:
                    ctle_gains = [gain/10 for gain in self.history['ctle_gain'][-100:]]  # Normalize for plotting
                    self.axes[1, 1].plot(times, ctle_gains, 'c-', label='CTLE Gain/10')
                
                self.axes[1, 1].set_title('ISI and CTLE Monitoring')
                self.axes[1, 1].set_ylabel('Ratio / Normalized Gain')
                self.axes[1, 1].legend()
                self.axes[1, 1].grid(True, alpha=0.3)
            
            # Add alert indicators
            recent_alerts = [alert for alert in self.history['alerts'] 
                           if alert['timestamp'] > times[0] if times]
            
            for alert in recent_alerts:
                alert_time = alert['timestamp']
                for ax in self.axes.flatten():
                    ax.axvline(x=alert_time, color='red', alpha=0.5, linestyle=':')
            
            plt.tight_layout()
            plt.pause(0.01)
            
        except Exception as e:
            self.logger.error(f"Plotting error: {e}")
    
    def start_monitoring(self):
        """Start real-time monitoring"""
        if self.running:
            self.logger.warning("Monitor already running")
            return
        
        self.running = True
        self.logger.info("Starting real-time Ethernet monitoring...")
        
        # Start worker threads
        self.analysis_thread = threading.Thread(target=self._analysis_worker)
        self.alert_thread = threading.Thread(target=self._alert_worker)
        
        self.analysis_thread.start()
        self.alert_thread.start()
        
        # Start plotting if available
        if HAS_PLOTTING:
            self.plotting_thread = threading.Thread(target=self._plotting_worker)
            self.plotting_thread.start()
        
        self.logger.info("Real-time monitoring started")
        print("\\n" + "="*60)
        print("REAL-TIME ETHERNET TESTING MONITOR ACTIVE")
        print("="*60)
        print(f"Signal Type: {self.config.signal_type}")
        print(f"Sample Rate: {self.config.sample_rate/1e9:.1f} GSa/s")
        print(f"Symbol Rate: {self.config.symbol_rate/1e9:.1f} GBaud")
        print(f"Analysis Interval: {self.config.analysis_interval*1000:.0f}ms")
        print("="*60)
    
    def _plotting_worker(self):
        """Worker thread for live plotting"""
        while self.running:
            try:
                self._update_live_plot()
                time.sleep(1.0)  # Update plots every second
            except Exception as e:
                self.logger.error(f"Plotting worker error: {e}")
                time.sleep(1.0)
    
    def stop_monitoring(self):
        """Stop real-time monitoring"""
        if not self.running:
            return
        
        self.logger.info("Stopping real-time monitoring...")
        self.running = False
        
        # Wait for threads to finish
        if self.analysis_thread:
            self.analysis_thread.join(timeout=5.0)
        if self.alert_thread:
            self.alert_thread.join(timeout=5.0)
        if self.plotting_thread:
            self.plotting_thread.join(timeout=5.0)
        
        self.logger.info("Real-time monitoring stopped")
        
        # Print summary
        print("\\n" + "="*60)
        print("MONITORING SESSION SUMMARY")
        print("="*60)
        print(f"Total samples processed: {self.samples_processed:,}")
        print(f"Total alerts generated: {self.total_alerts}")
        print(f"Average analysis time: {np.mean(self.analysis_time_history)*1000:.2f}ms")
        print("="*60)
    
    def get_status(self) -> Dict:
        """Get current monitoring status"""
        return {
            'running': self.running,
            'samples_processed': self.samples_processed,
            'total_alerts': self.total_alerts,
            'queue_size': self.signal_queue.qsize(),
            'alert_queue_size': self.alert_queue.qsize(),
            'avg_analysis_time_ms': np.mean(self.analysis_time_history) * 1000 if self.analysis_time_history else 0,
            'recent_alerts': self.history['alerts'][-10:] if self.history['alerts'] else []
        }

# Example integration functions for test equipment
def simulate_test_equipment_data():
    """Simulate data from Ethernet test equipment"""
    sample_rate = 25e9
    symbol_rate = 10e9
    
    while True:
        # Generate simulated signal data
        num_samples = 500
        time_vector = np.arange(num_samples) / sample_rate
        
        # PAM4 signal with some anomalies
        base_signal = np.random.choice([-0.75, -0.25, 0.25, 0.75], size=num_samples)
        noise = np.random.normal(0, 0.02, size=num_samples)
        signal = base_signal + noise
        
        # Occasionally inject anomalies
        if np.random.random() < 0.05:  # 5% chance
            # Eye closure anomaly
            signal *= 0.5  # Reduce eye height
        
        if np.random.random() < 0.03:  # 3% chance
            # Add excessive jitter
            jitter = np.random.normal(0, 0.1, size=num_samples)
            signal += jitter
        
        # Generate metadata
        metadata = {
            'ctle_gain': 5.0 + np.random.normal(0, 0.5),
            'tx_bits': np.random.choice([0, 1], size=100).tolist(),
            'rx_bits': np.random.choice([0, 1], size=100).tolist()
        }
        
        # Inject bit errors occasionally
        if np.random.random() < 0.1:  # 10% chance
            error_positions = np.random.choice(100, size=3, replace=False)
            for pos in error_positions:
                metadata['rx_bits'][pos] = 1 - metadata['rx_bits'][pos]
        
        yield time.time(), signal, metadata
        time.sleep(0.05)  # 20 Hz data rate

def main():
    """Demonstration of real-time Ethernet monitoring"""
    
    # Configuration for PAM4 testing
    config = EthernetTestConfig(
        signal_type="PAM4",
        sample_rate=25e9,
        symbol_rate=10e9,
        buffer_size=1000,
        analysis_interval=0.1,
        eye_height_min=1.0,
        eye_width_min=0.7,
        ber_threshold=1e-12,
        isi_threshold=0.1,
        enable_audio_alerts=True
    )
    
    # Create monitor
    monitor = RealTimeEthernetMonitor(config)
    
    try:
        # Start monitoring
        monitor.start_monitoring()
        
        # Simulate test equipment data feed
        print("\\nStarting simulated test equipment data feed...")
        print("Press Ctrl+C to stop monitoring\\n")
        
        data_generator = simulate_test_equipment_data()
        
        for timestamp, signal_data, metadata in data_generator:
            monitor.add_data_point(timestamp, signal_data, metadata)
            
            # Print periodic status
            if monitor.samples_processed % 10000 == 0 and monitor.samples_processed > 0:
                status = monitor.get_status()
                print(f"Status: {status['samples_processed']:,} samples, "
                     f"{status['total_alerts']} alerts, "
                     f"{status['avg_analysis_time_ms']:.1f}ms avg analysis time")
    
    except KeyboardInterrupt:
        print("\\nStopping monitoring...")
    
    finally:
        monitor.stop_monitoring()
        
        # Show final status
        status = monitor.get_status()
        print(f"\\nFinal Status:")
        print(f"  Samples processed: {status['samples_processed']:,}")
        print(f"  Total alerts: {status['total_alerts']}")
        print(f"  Average analysis time: {status['avg_analysis_time_ms']:.2f}ms")

if __name__ == "__main__":
    main()