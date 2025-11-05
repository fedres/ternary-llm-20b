#!/usr/bin/env python3
"""
Rich TUI Training Dashboard
A comprehensive real-time training monitoring interface
"""

import time
import random
import threading
import csv
import os
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass, field

from rich.console import Console
from rich.layout import Layout
from rich.panel import Panel
from rich.text import Text
from rich.table import Table
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn, SpinnerColumn
from rich.live import Live
from rich.prompt import Prompt
from rich.rule import Rule
from rich.traceback import install
from rich.align import Align
from rich.columns import Columns
from rich import box

# Install rich traceback handler
install()

console = Console()

@dataclass
class TrainingMetrics:
    """Data class for training metrics"""
    epoch: int = 0
    step: int = 0
    total_steps: int = 1000
    train_loss: float = 2.5
    val_loss: float = 2.8
    perplexity: float = 12.0
    learning_rate: float = 0.0001
    token_throughput: float = 2500.0
    batch_size: int = 32
    gpu_memory_used: float = 8.0
    gpu_memory_total: float = 12.0
    cpu_usage: float = 45.0
    gpu_usage: float = 85.0
    gpu_temp: float = 75.0
    ternary_bits_per_param: float = 2.0
    codebook_accuracy: float = 0.96
    kda_attention_ratio: float = 0.7
    training_time: float = 0.0
    paused: bool = False
    
    # Loss/accuracy history for charts
    train_loss_history: List[float] = field(default_factory=list)
    val_loss_history: List[float] = field(default_factory=list)
    val_perplexity_history: List[float] = field(default_factory=list)
    throughput_history: List[float] = field(default_factory=list)
    
    # System metrics history
    memory_history: List[float] = field(default_factory=list)
    gpu_usage_history: List[float] = field(default_factory=list)


class TrainingDashboard:
    """Main training dashboard with Rich TUI"""
    
    def __init__(self):
        self.console = Console()
        self.metrics = TrainingMetrics()
        self.layout = Layout()
        self.is_running = True
        self.start_time = time.time()
        self.data_file = "training_metrics.csv"
        
        # Initialize CSV file
        self._init_csv_file()
        
        # Setup layout
        self._setup_layout()
        
        # Setup keyboard listener
        self._setup_keyboard_listener()
        
    def _init_csv_file(self):
        """Initialize CSV file for metrics export"""
        if not os.path.exists(self.data_file):
            with open(self.data_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'timestamp', 'epoch', 'step', 'train_loss', 'val_loss', 'perplexity',
                    'learning_rate', 'token_throughput', 'gpu_memory', 'cpu_usage',
                    'gpu_usage', 'gpu_temp', 'ternary_bits', 'codebook_accuracy', 'kda_ratio'
                ])
    
    def _setup_layout(self):
        """Setup the multipanel layout"""
        self.layout.split(
            Layout(name="header", size=3),
            Layout(name="main", ratio=1),
            Layout(name="footer", size=3)
        )
        
        # Split main area into left and right
        self.layout["main"].split_row(
            Layout(name="left", ratio=2),
            Layout(name="right", ratio=1)
        )
        
        # Split left area vertically
        self.layout["left"].split(
            Layout(name="model_info", ratio=1),
            Layout(name="training_progress", ratio=1),
            Layout(name="metrics", ratio=1)
        )
        
        # Split right area vertically
        self.layout["right"].split(
            Layout(name="system_performance", ratio=1),
            Layout(name="charts", ratio=1)
        )
    
    def _setup_keyboard_listener(self):
        """Setup keyboard listener for interactive controls"""
        def keyboard_listener():
            import sys
            import select
            import tty
            import termios
            
            old_settings = termios.tcgetattr(sys.stdin)
            tty.setraw(sys.stdin.fileno())
            
            try:
                while self.is_running:
                    if select.select([sys.stdin], [], [], 0.1)[0]:
                        key = sys.stdin.read(1).lower()
                        self._handle_keypress(key)
            finally:
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
        
        keyboard_thread = threading.Thread(target=keyboard_listener, daemon=True)
        keyboard_thread.start()
    
    def _handle_keypress(self, key: str):
        """Handle keyboard input"""
        if key == 'q':
            self.is_running = False
        elif key == 'p':
            self.metrics.paused = not self.metrics.paused
        elif key == 's':
            self._save_checkpoint()
        elif key == 'l':
            self._adjust_learning_rate()
        elif key == 'b':
            self._adjust_batch_size()
        elif key == 'e':
            self._export_metrics()
        elif key == 'r':
            self._run_evaluation()
    
    def _save_checkpoint(self):
        """Save current training checkpoint"""
        checkpoint_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        console.log(f"[cyan]Saving checkpoint at epoch {self.metrics.epoch}, step {self.metrics.step}[/cyan]")
        # Simulate checkpoint save
        time.sleep(0.5)
    
    def _adjust_learning_rate(self):
        """Interactive learning rate adjustment"""
        new_lr = Prompt.ask(
            "Enter new learning rate",
            default=str(self.metrics.learning_rate)
        )
        try:
            self.metrics.learning_rate = float(new_lr)
            console.log(f"[green]Learning rate updated to {self.metrics.learning_rate}[/green]")
        except ValueError:
            console.log("[red]Invalid learning rate format[/red]")
    
    def _adjust_batch_size(self):
        """Interactive batch size adjustment"""
        new_batch = Prompt.ask(
            "Enter new batch size",
            default=str(self.metrics.batch_size)
        )
        try:
            self.metrics.batch_size = int(new_batch)
            console.log(f"[green]Batch size updated to {self.metrics.batch_size}[/green]")
        except ValueError:
            console.log("[red]Invalid batch size format[/red]")
    
    def _export_metrics(self):
        """Export current metrics to CSV"""
        console.log("[cyan]Exporting metrics to CSV...[/cyan]")
        # Metrics are already being exported in real-time
        console.log(f"[green]Metrics exported to {self.data_file}[/green]")
    
    def _run_evaluation(self):
        """Run model evaluation"""
        console.log("[cyan]Running model evaluation...[/cyan]")
        # Simulate evaluation
        eval_loss = random.uniform(2.0, 3.0)
        eval_perp = random.uniform(8.0, 15.0)
        console.log(f"[green]Evaluation complete - Loss: {eval_loss:.4f}, Perplexity: {eval_perp:.4f}[/green]")
    
    def _update_metrics(self):
        """Update simulated training metrics"""
        if self.metrics.paused:
            return
        
        # Simulate training progression
        self.metrics.step += 1
        if self.metrics.step % 50 == 0:
            self.metrics.epoch += 1
        
        # Update training metrics
        if not self.metrics.paused:
            self.metrics.train_loss = max(0.1, self.metrics.train_loss * (1 + random.uniform(-0.01, 0.005)))
            self.metrics.val_loss = max(0.1, self.metrics.val_loss * (1 + random.uniform(-0.008, 0.003)))
            self.metrics.perplexity = max(1.0, self.metrics.perplexity * (1 + random.uniform(-0.005, 0.002)))
            self.metrics.token_throughput = max(2000.0, self.metrics.token_throughput * (1 + random.uniform(-0.02, 0.03)))
            self.metrics.learning_rate *= 0.9995  # Gradual decay
            
            # Update system metrics
            self.metrics.gpu_memory_used = min(self.metrics.gpu_memory_total, 
                                             self.metrics.gpu_memory_used + random.uniform(-0.1, 0.2))
            self.metrics.gpu_usage = max(60.0, min(95.0, self.metrics.gpu_usage + random.uniform(-3.0, 2.0)))
            self.metrics.cpu_usage = max(20.0, min(80.0, self.metrics.cpu_usage + random.uniform(-2.0, 3.0)))
            self.metrics.gpu_temp = max(60.0, min(85.0, self.metrics.gpu_temp + random.uniform(-1.0, 1.0)))
            
            # Ternary quantization metrics
            self.metrics.ternary_bits_per_param = max(1.8, min(2.1, self.metrics.ternary_bits_per_param + random.uniform(-0.01, 0.005)))
            self.metrics.codebook_accuracy = max(0.94, min(0.98, self.metrics.codebook_accuracy + random.uniform(-0.005, 0.002)))
            self.metrics.kda_attention_ratio = max(0.6, min(0.8, self.metrics.kda_attention_ratio + random.uniform(-0.01, 0.005)))
            
            self.metrics.training_time = time.time() - self.start_time
        
        # Update histories (keep last 100 points)
        self.metrics.train_loss_history.append(self.metrics.train_loss)
        self.metrics.val_loss_history.append(self.metrics.val_loss)
        self.metrics.val_perplexity_history.append(self.metrics.perplexity)
        self.metrics.throughput_history.append(self.metrics.token_throughput)
        self.metrics.memory_history.append(self.metrics.gpu_memory_used / self.metrics.gpu_memory_total * 100)
        self.metrics.gpu_usage_history.append(self.metrics.gpu_usage)
        
        # Limit history length
        history_length = 100
        if len(self.metrics.train_loss_history) > history_length:
            self.metrics.train_loss_history = self.metrics.train_loss_history[-history_length:]
            self.metrics.val_loss_history = self.metrics.val_loss_history[-history_length:]
            self.metrics.val_perplexity_history = self.metrics.val_perplexity_history[-history_length:]
            self.metrics.throughput_history = self.metrics.throughput_history[-history_length:]
            self.metrics.memory_history = self.metrics.memory_history[-history_length:]
            self.metrics.gpu_usage_history = self.metrics.gpu_usage_history[-history_length:]
        
        # Export to CSV
        self._export_to_csv()
    
    def _export_to_csv(self):
        """Export current metrics to CSV file"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(self.data_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                timestamp,
                self.metrics.epoch,
                self.metrics.step,
                f"{self.metrics.train_loss:.4f}",
                f"{self.metrics.val_loss:.4f}",
                f"{self.metrics.perplexity:.4f}",
                f"{self.metrics.learning_rate:.6f}",
                f"{self.metrics.token_throughput:.1f}",
                f"{self.metrics.gpu_memory_used:.1f}",
                f"{self.metrics.cpu_usage:.1f}",
                f"{self.metrics.gpu_usage:.1f}",
                f"{self.metrics.gpu_temp:.1f}",
                f"{self.metrics.ternary_bits_per_param:.3f}",
                f"{self.metrics.codebook_accuracy:.4f}",
                f"{self.metrics.kda_attention_ratio:.4f}"
            ])
    
    def _render_header(self) -> Panel:
        """Render the header panel"""
        status = "⏸ PAUSED" if self.metrics.paused else "▶ RUNNING"
        status_color = "yellow" if self.metrics.paused else "green"
        
        header_text = Text()
        header_text.append("🤖 Training Dashboard", style="bold cyan")
        header_text.append(" | ", style="dim")
        header_text.append(f"Status: {status}", style=status_color)
        header_text.append(" | ", style="dim")
        header_text.append(f"Time: {datetime.now().strftime('%H:%M:%S')}", style="dim")
        
        return Panel(
            Align.center(header_text),
            style="bold",
            border_style="cyan"
        )
    
    def _render_model_info(self) -> Panel:
        """Render model information panel"""
        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_column("Key", style="cyan", no_wrap=True)
        table.add_column("Value", style="white")
        
        table.add_row("Model Architecture:", "Transformer-XL 1.3B")
        table.add_row("Parameters:", "1.3B")
        table.add_row("Quantization:", "Ternary (2-bit)")
        table.add_row("Attention:", "KDA + Full Hybrid")
        table.add_row("Optimizer:", "AdamW")
        table.add_row("Training Data:", "OpenWebText 40B tokens")
        table.add_row("Precision:", "Mixed (FP16/BF16)")
        table.add_row("Checkpoint Path:", "/models/checkpoints/")
        
        return Panel(
            table,
            title="📊 Model Information",
            border_style="purple",
            padding=(1, 2)
        )
    
    def _render_training_progress(self) -> Panel:
        """Render training progress panel"""
        progress_table = Table(show_header=False, box=None, padding=(0, 1))
        progress_table.add_column("Metric", style="cyan", no_wrap=True)
        progress_table.add_column("Current", style="white")
        progress_table.add_column("Target", style="dim")
        
        progress_table.add_row(
            "Epoch Progress",
            f"{self.metrics.step % 100}/100 steps",
            "100 steps"
        )
        
        progress_table.add_row(
            "Overall Progress", 
            f"{self.metrics.epoch}/{self.metrics.total_steps // 100} epochs",
            f"{self.metrics.total_steps // 100} epochs"
        )
        
        progress_table.add_row(
            "Training Time",
            f"{self.metrics.training_time // 3600:.0f}h {(self.metrics.training_time % 3600) // 60:.0f}m",
            "~24h estimated"
        )
        
        # Progress bar for epoch
        epoch_progress = (self.metrics.step % 100) / 100
        progress_bar = "█" * int(epoch_progress * 30) + "░" * (30 - int(epoch_progress * 30))
        
        progress_table.add_row("Epoch Progress", f"[{progress_bar}] {epoch_progress:.1%}", "")
        
        return Panel(
            progress_table,
            title="⏱️ Training Progress",
            border_style="blue",
            padding=(1, 2)
        )
    
    def _render_metrics(self) -> Panel:
        """Render metrics panel"""
        metrics_table = Table(show_header=False, box=None, padding=(0, 1))
        metrics_table.add_column("Metric", style="cyan", no_wrap=True)
        metrics_table.add_column("Value", style="white")
        metrics_table.add_column("Δ", style="dim")
        
        # Calculate deltas (simulated)
        loss_delta = random.uniform(-0.01, 0.01)
        perp_delta = random.uniform(-0.1, 0.1)
        lr_delta = random.uniform(-0.000001, 0.000001)
        
        loss_color = "green" if loss_delta < 0 else "red"
        perp_color = "green" if perp_delta < 0 else "red"
        lr_color = "dim"
        
        metrics_table.add_row("Training Loss", f"{self.metrics.train_loss:.4f}", f"[{loss_color}]{loss_delta:+.4f}[/{loss_color}]")
        metrics_table.add_row("Validation Loss", f"{self.metrics.val_loss:.4f}", f"[{loss_color}]{loss_delta:+.4f}[/{loss_color}]")
        metrics_table.add_row("Perplexity", f"{self.metrics.perplexity:.2f}", f"[{perp_color}]{perp_delta:+.2f}[/{perp_color}]")
        metrics_table.add_row("Learning Rate", f"{self.metrics.learning_rate:.6f}", f"[{lr_color}]{lr_delta:+.6f}[/{lr_color}]")
        metrics_table.add_row("Token Throughput", f"{self.metrics.token_throughput:.0f} tok/s", "+125 tok/s")
        metrics_table.add_row("Batch Size", f"{self.metrics.batch_size}", "")
        
        # Quantization metrics
        metrics_table.add_row("Ternary Bits/Param", f"{self.metrics.ternary_bits_per_param:.3f}", "+0.001")
        metrics_table.add_row("Codebook Accuracy", f"{self.metrics.codebook_accuracy:.4f}", "+0.002")
        metrics_table.add_row("KDA Ratio", f"{self.metrics.kda_attention_ratio:.2%}", "+1.2%")
        
        return Panel(
            metrics_table,
            title="📈 Training Metrics",
            border_style="green",
            padding=(1, 2)
        )
    
    def _render_system_performance(self) -> Panel:
        """Render system performance panel"""
        perf_table = Table(show_header=False, box=None, padding=(0, 1))
        perf_table.add_column("Component", style="cyan", no_wrap=True)
        perf_table.add_column("Usage", style="white")
        perf_table.add_column("Status", style="dim")
        
        # GPU Memory
        gpu_mem_percent = (self.metrics.gpu_memory_used / self.metrics.gpu_memory_total) * 100
        gpu_mem_color = "red" if gpu_mem_percent > 85 else "yellow" if gpu_mem_percent > 70 else "green"
        perf_table.add_row(
            "GPU Memory",
            f"{self.metrics.gpu_memory_used:.1f}GB / {self.metrics.gpu_memory_total:.1f}GB ({gpu_mem_percent:.1f}%)",
            f"[{gpu_mem_color}]●[/]"
        )
        
        # GPU Usage
        gpu_color = "green" if self.metrics.gpu_usage < 90 else "red"
        perf_table.add_row(
            "GPU Usage",
            f"{self.metrics.gpu_usage:.1f}%",
            f"[{gpu_color}]●[/]"
        )
        
        # GPU Temperature
        temp_color = "red" if self.metrics.gpu_temp > 80 else "yellow" if self.metrics.gpu_temp > 75 else "green"
        perf_table.add_row(
            "GPU Temperature",
            f"{self.metrics.gpu_temp:.0f}°C",
            f"[{temp_color}]●[/]"
        )
        
        # CPU Usage
        cpu_color = "green" if self.metrics.cpu_usage < 70 else "yellow" if self.metrics.cpu_usage < 85 else "red"
        perf_table.add_row(
            "CPU Usage",
            f"{self.metrics.cpu_usage:.1f}%",
            f"[{cpu_color}]●[/]"
        )
        
        return Panel(
            perf_table,
            title="💻 System Performance",
            border_style="magenta",
            padding=(1, 2)
        )
    
    def _render_charts(self) -> Panel:
        """Render charts panel using text-based visualizations"""
        
        # Create loss history visualization
        if len(self.metrics.train_loss_history) > 1:
            # Loss history table
            loss_table = Table(title="Loss History (Last 20 Steps)", show_header=True, box=None, pad_edge=False)
            loss_table.add_column("Step", style="cyan", width=8)
            loss_table.add_column("Train Loss", style="white", width=12)
            loss_table.add_column("Val Loss", style="white", width=12)
            loss_table.add_column("Trend", style="dim", width=12)
            
            # Show last 10 data points
            recent_losses = list(zip(
                self.metrics.train_loss_history[-10:], 
                self.metrics.val_loss_history[-10:]
            ))
            
            for i, (train_loss, val_loss) in enumerate(recent_losses[-10:]):
                step = len(self.metrics.train_loss_history) - 10 + i
                trend = ""
                if i > 0:
                    prev_train = recent_losses[i-1][0]
                    prev_val = recent_losses[i-1][1]
                    if train_loss < prev_train:
                        trend = "📉 improving"
                    elif train_loss > prev_train:
                        trend = "📈 worsening"
                    else:
                        trend = "➡️ stable"
                
                loss_table.add_row(
                    str(step),
                    f"{train_loss:.4f}",
                    f"{val_loss:.4f}",
                    trend
                )
        else:
            loss_table = Text("Collecting loss data...", style="dim", justify="center")
        
        # Create throughput visualization
        if len(self.metrics.throughput_history) > 1:
            # Throughput trend table
            throughput_table = Table(title="Token Throughput (Last 15 Steps)", show_header=True, box=None, pad_edge=False)
            throughput_table.add_column("Step", style="cyan", width=8)
            throughput_table.add_column("Throughput", style="white", width=12)
            throughput_table.add_column("Performance", style="dim", width=15)
            
            recent_throughput = self.metrics.throughput_history[-15:]
            avg_throughput = sum(recent_throughput) / len(recent_throughput)
            
            for i, throughput in enumerate(recent_throughput[-10:]):
                step = len(self.metrics.throughput_history) - 10 + i
                performance = ""
                if throughput >= 2500:
                    performance = "🟢 Excellent"
                elif throughput >= 2000:
                    performance = "🟡 Good"
                else:
                    performance = "🔴 Needs work"
                
                throughput_table.add_row(
                    str(step),
                    f"{throughput:.0f} tok/s",
                    performance
                )
            
            # Add summary
            throughput_table.add_row("─" * 8, "─" * 12, "─" * 15)
            throughput_table.add_row(
                "Average",
                f"{avg_throughput:.0f} tok/s",
                f"Target: 2,500 tok/s"
            )
        else:
            throughput_table = Text("Collecting throughput data...", style="dim", justify="center")
        
        # Create combined chart container
        chart_layout = Layout()
        chart_layout.split_column(
            Layout(loss_table, ratio=1),
            Layout(throughput_table, ratio=1)
        )
        
        return Panel(
            chart_layout,
            title="📊 Performance Trends",
            border_style="yellow",
            padding=(1, 1)
        )
    
    def _render_footer(self) -> Panel:
        """Render footer with controls"""
        controls = Text()
        controls.append("[bold]Controls:[/bold] ")
        controls.append("[cyan]Q[/cyan]=Quit ")
        controls.append("[cyan]P[/cyan]=Pause/Resume ")
        controls.append("[cyan]S[/cyan]=Save Checkpoint ")
        controls.append("[cyan]L[/cyan]=Adjust LR ")
        controls.append("[cyan]B[/cyan]=Adjust Batch ")
        controls.append("[cyan]E[/cyan]=Export CSV ")
        controls.append("[cyan]R[/cyan]=Run Eval ")
        
        stats = Text()
        stats.append(f" | Steps: {self.metrics.step:,} | ")
        stats.append(f"Memory: {self.metrics.gpu_memory_used:.1f}GB | ")
        stats.append(f"Throughput: {self.metrics.token_throughput:.0f} tok/s | ")
        stats.append(f"Target: 2,500+ tok/s")
        
        footer_content = Align.center(controls + stats)
        
        return Panel(
            footer_content,
            style="dim",
            border_style="dim",
            padding=(0, 2)
        )
    
    def _render_dashboard(self) -> str:
        """Render the complete dashboard"""
        self._update_metrics()
        
        # Update layout with current panels
        self.layout["header"].update(self._render_header())
        self.layout["model_info"].update(self._render_model_info())
        self.layout["training_progress"].update(self._render_training_progress())
        self.layout["metrics"].update(self._render_metrics())
        self.layout["system_performance"].update(self._render_system_performance())
        self.layout["charts"].update(self._render_charts())
        self.layout["footer"].update(self._render_footer())
        
        return self.layout
    
    def run(self):
        """Run the training dashboard"""
        try:
            with Live(self._render_dashboard(), refresh_per_second=4, screen=True) as live:
                while self.is_running:
                    live.update(self._render_dashboard())
                    time.sleep(0.25)  # 4 FPS update rate
        except KeyboardInterrupt:
            console.log("\n[yellow]Dashboard interrupted by user[/yellow]")
        except Exception as e:
            console.log(f"[red]Error in dashboard: {e}[/red]")
        finally:
            self.is_running = False
            console.log("[cyan]Dashboard closed[/cyan]")


def main():
    """Main function to run the training dashboard"""
    console.print("[bold cyan]🚀 Starting Training Dashboard[/bold cyan]")
    console.print("[dim]Press 'q' to quit, 'p' to pause/resume, or '?' for help[/dim]\n")
    
    dashboard = TrainingDashboard()
    dashboard.run()


if __name__ == "__main__":
    main()