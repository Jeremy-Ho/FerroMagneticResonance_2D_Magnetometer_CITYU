import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg backend for real-time updates
import time
import os


# Import custom device libraries
from Core_XY_Platform import CoreXY_Platform
from vector_network_analyzer_NN import VectorNetworkAnalyzer
    


# Set matplotlib parameters for optimal performance
plt.rcParams['path.simplify'] = True
plt.rcParams['path.simplify_threshold'] = 1.0
plt.rcParams['agg.path.chunksize'] = 10000

class LiveFrequencyMonitor:
    """Lightweight live visualization for CoreXY + VNA experiment"""
    
    def __init__(self, x_points, y_points, start_freq_ghz, stop_freq_ghz):
        """
        Initialize the live visualization
        
        Parameters:
        x_points: Number of X-axis measurement points
        y_points: Number of Y-axis measurement points
        start_freq_ghz: Start frequency in GHz
        stop_freq_ghz: Stop frequency in GHz
        """
        self.x_points = x_points
        self.y_points = y_points
        self.start_freq_ghz = start_freq_ghz
        self.stop_freq_ghz = stop_freq_ghz
        self.freq_range_ghz = stop_freq_ghz - start_freq_ghz
        
        # Initialize data arrays - set initial value to middle of frequency range
        initial_freq = (start_freq_ghz + stop_freq_ghz) / 2
        self.peak_frequencies = np.full((y_points, x_points), initial_freq, dtype=np.float32)
        
        # Create figure with two subplots
        self.fig, (self.ax_map, self.ax_spectrum) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Setup 2D Frequency Map
        self.setup_frequency_map()
        
        # Setup Spectrum Plot
        self.setup_spectrum_plot()
        
        # Add progress text
        self.progress_text = self.ax_map.text(
            0.02, 0.02, 'Initializing...', 
            transform=self.ax_map.transAxes,
            fontsize=10,
            verticalalignment='bottom',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
        )
        
        plt.tight_layout()
        plt.ion()  # Enable interactive mode
        plt.show(block=False)
        
        # Performance optimization
        self.last_update_time = time.time()
        self.update_interval = 0.5  # Minimum time between updates (seconds)
        
        print(f"Live visualization initialized for frequency range: {start_freq_ghz:.2f} - {stop_freq_ghz:.2f} GHz")
        print("Please wait for plot window...")
        plt.pause(0.5)  # Allow window to open
        
    def setup_frequency_map(self):
        """Configure the 2D frequency map plot"""
        # Create heatmap with initial values
        self.heatmap = self.ax_map.imshow(
            self.peak_frequencies,
            cmap='grey',
            aspect='auto',
            origin='upper',
            vmin=self.start_freq_ghz,   # Dynamic start frequency
            vmax=self.stop_freq_ghz     # Dynamic stop frequency
        )
        
        # Add colorbar
        cbar = self.fig.colorbar(self.heatmap, ax=self.ax_map, fraction=0.046, pad=0.04)
        cbar.set_label('Peak Frequency (GHz)')
        
        # Labels and title
        self.ax_map.set_xlabel('X Position')
        self.ax_map.set_ylabel('Y Position')
        self.ax_map.set_title(f'Live Frequency Map\n{self.start_freq_ghz:.2f} - {self.stop_freq_ghz:.2f} GHz')
        
        # Add grid (optional, can be removed for better performance)
        self.ax_map.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
        
    def setup_spectrum_plot(self):
        """Configure the live spectrum plot"""
        # Initialize empty plot
        self.spectrum_line, = self.ax_spectrum.plot([], [], 'b-', linewidth=1.5, alpha=0.8)
        
        # Mark the peak frequency
        self.peak_marker = self.ax_spectrum.scatter([], [], c='red', s=40, zorder=5)
        
        # Configure plot
        self.ax_spectrum.set_xlabel('Frequency (GHz)')
        self.ax_spectrum.set_ylabel('Transmission')
        self.ax_spectrum.set_title('Live Spectrum')
        self.ax_spectrum.grid(True, alpha=0.3, linestyle=':')
        
        # Set dynamic axis limits based on frequency range
        self.ax_spectrum.set_xlim(self.start_freq_ghz, self.stop_freq_ghz)
        
        # Set initial Y-axis limits (will auto-adjust during measurements)
        # Common VNA range, will adjust automatically
        self.ax_spectrum.set_ylim(-60, 0)  
        
        # Add text box for current measurement info
        self.spectrum_info = self.ax_spectrum.text(
            0.05, 0.95, f'Frequency Range: {self.start_freq_ghz:.2f} - {self.stop_freq_ghz:.2f} GHz',
            transform=self.ax_spectrum.transAxes,
            fontsize=9,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7)
        )
        
    def update_display(self, x_idx, y_idx, spectrum_data, peak_freq_hz):
        """
        Update the live display with new measurement data
        
        Parameters:
        x_idx: Current X position index (0-based)
        y_idx: Current Y position index (0-based)
        spectrum_data: 1D array of transmission values
        peak_freq_hz: Peak frequency in Hz
        """
        # Throttle updates for performance
        current_time = time.time()
        if current_time - self.last_update_time < self.update_interval:
            return
        
        # Convert peak frequency to GHz for display
        peak_freq_ghz = peak_freq_hz / 1e9
        
        # Update frequency map
        self.peak_frequencies[y_idx, x_idx] = peak_freq_ghz
        self.heatmap.set_data(self.peak_frequencies)
        
        # Optional: Auto-adjust color scale for better contrast
        # Comment out for fixed scale (between start and stop frequencies)
        measured_min = np.min(self.peak_frequencies[self.peak_frequencies > 0])
        measured_max = np.max(self.peak_frequencies)
        margin = 0.01 * self.freq_range_ghz  # 1% margin
        self.heatmap.set_clim(vmin=max(self.start_freq_ghz, measured_min - margin),
                             vmax=min(self.stop_freq_ghz, measured_max + margin))
        
        # Update spectrum plot
        if len(spectrum_data) > 0:
            # Generate frequency axis based on current frequency range
            freq_axis = np.linspace(self.start_freq_ghz, self.stop_freq_ghz, len(spectrum_data))
            self.spectrum_line.set_data(freq_axis, spectrum_data)
            
            # Find the actual peak in the spectrum (min for absorption dip)
            if len(spectrum_data) > 0:
                actual_peak_idx = np.argmin(spectrum_data)
                actual_peak_freq_ghz = freq_axis[actual_peak_idx]
                actual_peak_value = spectrum_data[actual_peak_idx]
                
                # Update peak marker
                self.peak_marker.set_offsets([[actual_peak_freq_ghz, actual_peak_value]])
            
            # Auto-scale Y axis for spectrum (with margins)
            y_min, y_max = np.min(spectrum_data), np.max(spectrum_data)
            y_margin = (y_max - y_min) * 0.1 if (y_max - y_min) > 0 else 5
            self.ax_spectrum.set_ylim(y_min - y_margin, y_max + y_margin)
        
        # Update progress text
        total_points = self.x_points * self.y_points
        completed = y_idx * self.x_points + x_idx + 1
        progress_percent = (completed / total_points) * 100
        
        progress_text = f"Progress: {progress_percent:.1f}%\n"
        progress_text += f"Completed: {completed}/{total_points}\n"
        progress_text += f"Current: X={x_idx}, Y={y_idx}\n"
        progress_text += f"Peak: {peak_freq_ghz:.4f} GHz"
        self.progress_text.set_text(progress_text)
        
        # Update spectrum info
        spectrum_info = f"Current Spectrum\n"
        spectrum_info += f"Range: {self.start_freq_ghz:.2f}-{self.stop_freq_ghz:.2f} GHz\n"
        spectrum_info += f"Min: {np.min(spectrum_data):.2f}\n"
        spectrum_info += f"Max: {np.max(spectrum_data):.2f}"
        self.spectrum_info.set_text(spectrum_info)
        
        # Perform efficient update
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        
        # Update timing
        self.last_update_time = current_time
        
    def finalize(self):
        """Final update and switch to blocking mode"""
        print("\nExperiment complete! Finalizing display...")
        
        # One final full update
        self.fig.canvas.draw()
        
        # Switch to blocking mode for final viewing
        plt.ioff()
        plt.show()
        
    def save_data(self, save_path):
        """Save the frequency map data to file"""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Save as numpy array
        np.save(save_path, self.peak_frequencies)
        
        # Also save as text file for easy viewing
        txt_path = save_path.replace('.npy', '.txt')
        np.savetxt(txt_path, self.peak_frequencies, fmt='%.6f', delimiter=',')
        
        print(f"Frequency map data saved to:\n  {save_path}\n  {txt_path}")

def run_experiment_with_live_monitor(PORT_VNA="COM3"):
    """
    Main experiment function with integrated live visualization
    
    This assumes you have the CoreXY_Platform and VectorNetworkAnalyzer
    classes available as in your original code
    """
    
    # Initialize devices
    print("Initializing devices...")
    vna = VectorNetworkAnalyzer()
    vna.connect("COM3")  # Replace with appropriate COM port
    platform = CoreXY_Platform()
    
    try:
        # User input for experiment parameters
        print("\n" + "="*50)
        print("EXPERIMENT SETUP")
        print("="*50)
        
        # Get frequency parameters (in GHz)
        print("\n--- Frequency Configuration ---")
        start_freq_ghz = float(input("Enter start frequency (GHz): "))
        stop_freq_ghz = float(input("Enter stop frequency (GHz): "))
        
        # Validate frequency range
        if stop_freq_ghz <= start_freq_ghz:
            raise ValueError("Stop frequency must be greater than start frequency")
        
        # Convert to Hz for VNA
        start_freq_hz = int(start_freq_ghz * 1e9)
        stop_freq_hz = int(stop_freq_ghz * 1e9)
        
        # Get VNA points
        vna_points = int(input("Enter number of VNA frequency points: "))
        
        # Get position parameters
        print("\n--- Position Configuration ---")
        x_start = float(input("Enter starting X position (mm): "))
        x_end = float(input("Enter ending X position (mm): "))
        x_points = int(input("Number of X points: "))
        
        y_start = float(input("Enter starting Y position (mm): "))
        y_end = float(input("Enter ending Y position (mm): "))
        y_points = int(input("Number of Y points: "))
        
        # Validate inputs
        if x_end <= x_start:
            raise ValueError("X end must be greater than X start")
        if y_end <= y_start:
            raise ValueError("Y end must be greater than Y start")
        if x_points <= 0 or y_points <= 0 or vna_points <= 0:
            raise ValueError("Number of points must be positive")
        
        # Calculate step sizes
        x_step = (x_end - x_start) / (x_points - 1) if x_points > 1 else 0
        y_step = (y_end - y_start) / (y_points - 1) if y_points > 1 else 0
        
        print(f"\nScan parameters:")
        print(f"  Frequency: {start_freq_ghz:.3f} to {stop_freq_ghz:.3f} GHz ({vna_points} points)")
        print(f"  X: {x_start:.2f} to {x_end:.2f} mm ({x_points} points, step: {x_step:.3f} mm)")
        print(f"  Y: {y_start:.2f} to {y_end:.2f} mm ({y_points} points, step: {y_step:.3f} mm)")
        print(f"  Total measurements: {x_points * y_points}")

        #Experiment physical parameter input
        Extension_Len = str(input("Extension used: "))
        Experiment_description = str(input("Describe experiment: "))

        # Initialize live monitor
        print("\nInitializing live visualization...")
        monitor = LiveFrequencyMonitor(x_points, y_points, start_freq_ghz, stop_freq_ghz)
        
        # Create output directory
        timestamp = time.strftime("%Y%m%d_%H%M")
        output_dir = f"C:\\School Code\\{time.strftime("%Y%m%d",time.localtime())}Experiment\\{timestamp}_Experiment_data"

        os.makedirs(output_dir, exist_ok=True)
        
        # Save experiment parameters
        with open(os.path.join(output_dir, "experiment_params.txt"), "w") as f:
            f.write(f"Experiment Time: {timestamp}\n")
            f.write(f"Start Frequency: {start_freq_hz} Hz ({start_freq_ghz:.3f} GHz)\n")
            f.write(f"Stop Frequency: {stop_freq_hz} Hz ({stop_freq_ghz:.3f} GHz)\n")
            f.write(f"VNA Points: {vna_points}\n")
            f.write(f"X Range: {x_start:.3f} to {x_end:.3f} mm\n")
            f.write(f"Y Range: {y_start:.3f} to {y_end:.3f} mm\n")
            f.write(f"X Points: {x_points}\n")
            f.write(f"Y Points: {y_points}\n")
            f.write(f"Extension used: {Extension_Len}\n")
            f.write(f"Experiment: {Experiment_description}\n")
        
        # Home the platform
        print("\nHoming platform...")
        platform.Manuel_Home()
        
        # Generate position arrays
        x_positions = np.linspace(x_start, x_end, x_points)
        y_positions = np.linspace(y_start, y_end, y_points)
        
        # Main measurement loop
        print("\n" + "="*50)
        print("STARTING MEASUREMENTS")
        print("="*50)
        
        start_time = time.time()
        
        for y_idx, y_pos in enumerate(y_positions):
            for x_idx, x_pos in enumerate(x_positions):
                measurement_start = time.time()
                measurement_num = y_idx * x_points + x_idx + 1
                total_measurements = x_points * y_points
                
                print(f"\n[{measurement_num}/{total_measurements}] ", end="")
                print(f"Moving to X={x_pos:.2f}, Y={y_pos:.2f} mm...")
                
                # Move platform
                platform.Move_to_Pos(x_pos, y_pos)
                
                # Measure spectrum
                print(f"  Measuring spectrum from {start_freq_ghz:.3f} to {stop_freq_ghz:.3f} GHz...")
                vna.measure_s_parameter(start_freq_hz, stop_freq_hz, vna_points, is_for_preparation=False)
                spectrum = vna.transmission_1darray
                
                # Find absorption dip (minimum)
                freq_array = np.linspace(start_freq_hz, stop_freq_hz, vna_points)
                dip_idx = np.argmin(spectrum)
                dip_freq = freq_array[dip_idx]
                
                # Save raw spectrum data
                filename = f"Position_X_{x_idx:.2f}_Y_{y_idx:.2f}.txt"
                filepath = os.path.join(output_dir, filename)
                np.savetxt(filepath, spectrum, delimiter=',')
                
                # Update live display
                monitor.update_display(x_idx, y_idx, spectrum, dip_freq)
                
                # Calculate and display timing
                measurement_time = time.time() - measurement_start
                print(f"  Dip frequency: {dip_freq/1e9:.6f} GHz")
                print(f"  Measurement time: {measurement_time:.1f}s")
                
                # Estimate remaining time
                completed = measurement_num
                elapsed = time.time() - start_time
                avg_time = elapsed / completed
                remaining = avg_time * (total_measurements - completed)
                
                if measurement_num % 5 == 0:  # Display ETA every 5 measurements
                    print(f"  ETA: {remaining/60:.1f} minutes remaining")
        
        # Experiment complete
        total_time = time.time() - start_time
        print("\n" + "="*50)
        print("EXPERIMENT COMPLETE!")
        print("="*50)
        print(f"Total time: {total_time/60:.1f} minutes")
        print(f"Average time per measurement: {total_time/(x_points*y_points):.1f}s")
        
        # Save final data
        monitor.save_data(os.path.join(output_dir, "frequency_map.npy"))

        # Generate summary plot
        print("\nGenerating summary plot...")
        fig_summary, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Final frequency map
        im1 = ax1.imshow(monitor.peak_frequencies, cmap='grey', aspect='auto', origin='upper')
        ax1.set_title(f'Final Frequency Map\n{start_freq_ghz:.3f} - {stop_freq_ghz:.3f} GHz')
        ax1.set_xlabel('X Position')
        ax1.set_ylabel('Y Position')
        plt.colorbar(im1, ax=ax1, label='Frequency (GHz)')
        
        # Histogram of peak frequencies
        ax2.hist(monitor.peak_frequencies.flatten(), bins=30, edgecolor='black', alpha=0.7)
        ax2.set_title('Distribution of Peak Frequencies')
        ax2.set_xlabel('Frequency (GHz)')
        ax2.set_ylabel('Count')
        ax2.grid(True, alpha=0.3)
        ax2.axvline(x=start_freq_ghz, color='red', linestyle='--', alpha=0.5, label=f'Start: {start_freq_ghz:.3f} GHz')
        ax2.axvline(x=stop_freq_ghz, color='blue', linestyle='--', alpha=0.5, label=f'Stop: {stop_freq_ghz:.3f} GHz')
        ax2.legend()
        
        plt.tight_layout()
        summary_path = os.path.join(output_dir, "summary_plot.png")
        plt.savefig(summary_path, dpi=150, bbox_inches='tight')
        print(f"Summary plot saved to: {summary_path}")
        
        # Finalize display
        monitor.finalize()
        
        print(f"\nAll data saved to: {os.path.abspath(output_dir)}")
        
    except KeyboardInterrupt:
        print("\n\nExperiment interrupted by user!")
        print("Saving current data...")
        
        # Try to save whatever data we have
        try:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            save_dir = f"interrupted_experiment_{timestamp}"
            os.makedirs(save_dir, exist_ok=True)
            
            # Save current frequency map
            if 'monitor' in locals():
                monitor.save_data(os.path.join(save_dir, "partial_frequency_map.npy"))
                
            print(f"Partial data saved to: {save_dir}")
        except:
            print("Could not save partial data")
            
    except Exception as e:
        print(f"\nError during experiment: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        # Cleanup
        print("\nCleaning up devices...")
        try:
            vna.disconnect()
        except:
            pass
            
        try:
            platform.Shutdown()
        except:
            pass
            
        print("Done!")

# Simple test mode without actual hardware
def test_mode():
    """Test the visualization without real hardware"""
    print("Running in TEST MODE (no hardware required)")
    
    # Get frequency parameters for test
    print("\n--- Test Mode Configuration ---")
    start_freq_ghz = float(input("Enter start frequency (GHz) [e.g., 2.2]: "))
    stop_freq_ghz = float(input("Enter stop frequency (GHz) [e.g., 2.6]: "))
    
    if stop_freq_ghz <= start_freq_ghz:
        print("Error: Stop frequency must be greater than start frequency")
        return
    
    # Simulate experiment parameters
    x_points = 20
    y_points = 70
    
    # Initialize monitor
    monitor = LiveFrequencyMonitor(x_points, y_points, start_freq_ghz, stop_freq_ghz)
    
    print(f"\nSimulating {x_points * y_points} measurements...")
    
    # Simulate measurements
    for y_idx in range(y_points):
        for x_idx in range(x_points):
            # Generate simulated spectrum (absorption dip around random frequency)
            # Dip frequency randomly within the specified range
            center_freq = start_freq_ghz + np.random.random() * (stop_freq_ghz - start_freq_ghz)
            spectrum_points = 151
            
            # Create spectrum with Gaussian absorption dip
            freq_axis = np.linspace(start_freq_ghz, stop_freq_ghz, spectrum_points)
            
            # Create realistic-looking spectrum with noise
            spectrum = -20 * np.exp(-((freq_axis - center_freq) / 0.03)**2)  # Gaussian dip
            spectrum += -10  # Baseline
            spectrum += np.random.normal(0, 0.3, spectrum_points)  # Noise
            
            # Update display
            monitor.update_display(x_idx, y_idx, spectrum, center_freq * 1e9)
            
            # Pause to simulate measurement time
            time.sleep(0.05)
    
    print("\nTest complete!")
    monitor.finalize()

if __name__ == "__main__":
    print("="*60)
    print("CoreXY + VNA Live Frequency Monitoring System")
    print("="*60)
    print("\nOptions:")
    print("  1. Run full experiment with hardware")
    print("  2. Test visualization only (no hardware)")
    
    choice = input("\nEnter choice (1 or 2): ").strip()
    
    if choice == "2":
        test_mode()
    else:
        # Check for required libraries
        try:
            from Core_XY_Platform import CoreXY_Platform
            from vector_network_analyzer_NN import VectorNetworkAnalyzer
            run_experiment_with_live_monitor(PORT_VNA="COM3")  # Replace with appropriate COM port
        except ImportError as e:
            print(f"\nError: Required library not found: {e}")
