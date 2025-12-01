"""
Interactive CLI Testing Tool for RUKO Multi-Agent System

Features:
- Interactive prompt for testing queries
- Extensive logging of all agent steps
- Timing information for each operation
- Agent registry inspection
- Session statistics

Commands:
  /quit or /exit  - Exit the CLI
  /agents         - List all registered agents
  /stats          - Show session statistics
  /clear          - Clear screen
  /verbose on/off - Toggle verbose mode
  /help           - Show help
"""
import asyncio
import sys
import os
import time
from datetime import datetime
from typing import List, Dict, Any
from dataclasses import dataclass, field

# Fix Windows encoding
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()

# ANSI color codes for terminal output
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    DIM = '\033[2m'


def colored(text: str, color: str) -> str:
    """Apply color to text"""
    return f"{color}{text}{Colors.ENDC}"


def print_header(text: str):
    """Print a header line"""
    print(colored(f"\n{'='*70}", Colors.CYAN))
    print(colored(f"  {text}", Colors.CYAN + Colors.BOLD))
    print(colored(f"{'='*70}", Colors.CYAN))


def print_section(title: str):
    """Print a section header"""
    print(colored(f"\n--- {title} ---", Colors.YELLOW))


def print_step(step: str, details: str = ""):
    """Print a step in the process"""
    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    print(colored(f"[{timestamp}]", Colors.DIM) + colored(f" >> {step}", Colors.GREEN) + (f" {details}" if details else ""))


def print_agent(agent_name: str, message: str):
    """Print agent-specific message"""
    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    agent_color = {
        "ORCHESTRATOR": Colors.BLUE,
        "SQL": Colors.GREEN,
        "PINECONE": Colors.CYAN,
        "WEB_SEARCH": Colors.YELLOW,
        "REVIEWER": Colors.HEADER,
        "AGENT_SYSTEM": Colors.BOLD,
    }.get(agent_name, Colors.ENDC)
    print(colored(f"[{timestamp}]", Colors.DIM) + colored(f" [{agent_name}]", agent_color) + f" {message}")


def print_error(message: str):
    """Print error message"""
    print(colored(f"ERROR: {message}", Colors.RED))


def print_success(message: str):
    """Print success message"""
    print(colored(f"SUCCESS: {message}", Colors.GREEN))


def print_info(message: str):
    """Print info message"""
    print(colored(f"INFO: {message}", Colors.BLUE))


@dataclass
class SessionStats:
    """Track session statistics"""
    queries_total: int = 0
    queries_success: int = 0
    queries_failed: int = 0
    total_time_ms: int = 0
    agents_invoked: Dict[str, int] = field(default_factory=dict)
    start_time: float = field(default_factory=time.time)

    def add_query(self, success: bool, time_ms: int, agents: List[str]):
        self.queries_total += 1
        if success:
            self.queries_success += 1
        else:
            self.queries_failed += 1
        self.total_time_ms += time_ms
        for agent in agents:
            self.agents_invoked[agent] = self.agents_invoked.get(agent, 0) + 1

    def avg_time_ms(self) -> float:
        if self.queries_total == 0:
            return 0
        return self.total_time_ms / self.queries_total

    def session_duration(self) -> str:
        duration = time.time() - self.start_time
        minutes = int(duration // 60)
        seconds = int(duration % 60)
        return f"{minutes}m {seconds}s"


class VerboseAgentSystem:
    """Wrapper around AgentSystem with enhanced logging"""

    def __init__(self):
        from rag.agents import create_agent_system, AgentRegistry

        print_step("Initializing Agent System...")

        # Create system with verbose mode
        self.system = create_agent_system(verbose=True)
        self.registry = AgentRegistry

        print_success("Agent System initialized")
        self._print_registered_agents()

    def _print_registered_agents(self):
        """Print all registered agents"""
        agents = self.registry.get_all_agents()
        print_section(f"Registered Agents ({len(agents)})")

        for agent_id, metadata in agents.items():
            reasoning_tag = colored("[REASONING]", Colors.YELLOW) if metadata.uses_reasoning else colored("[FAST]", Colors.GREEN)
            invokable_tag = colored("[INVOKABLE]", Colors.CYAN) if metadata.direct_invocation else colored("[INTERNAL]", Colors.DIM)

            print(f"  {colored(metadata.name, Colors.BOLD)}")
            print(f"    ID: {agent_id} {reasoning_tag} {invokable_tag}")
            print(f"    {metadata.description}")
            if metadata.default_model:
                print(f"    Model: {metadata.default_model}")
            print()

    async def process_query(self, query: str) -> Dict[str, Any]:
        """Process a query with extensive logging"""
        print_header(f"PROCESSING QUERY")
        print(f"\n{colored('Query:', Colors.BOLD)} {query}\n")

        start_time = time.time()

        # Track timing for each phase
        timings = {}

        # Phase 1: Context Setup
        phase_start = time.time()
        print_step("Phase 1: Context Setup")
        timings['context_setup'] = int((time.time() - phase_start) * 1000)

        # Process through the system
        result = await self.system.process(
            user_query=query,
            user_id='cli_test_user',
            user_name='CLI Tester'
        )

        total_time = int((time.time() - start_time) * 1000)

        # Print results
        print_section("RESULTS")
        print(f"  {colored('Success:', Colors.BOLD)} {colored('Yes', Colors.GREEN) if result.success else colored('No', Colors.RED)}")
        print(f"  {colored('Query Intent:', Colors.BOLD)} {result.query_intent or 'Unknown'}")
        print(f"  {colored('Agents Used:', Colors.BOLD)} {' -> '.join(result.agents_used)}")
        print(f"  {colored('Execution Time:', Colors.BOLD)} {result.execution_time_ms}ms")

        if result.error:
            print(f"  {colored('Error:', Colors.RED)} {result.error}")

        # Print metadata
        if result.metadata:
            print_section("METADATA")
            for key, value in result.metadata.items():
                print(f"  {key}: {value}")

        # Print response
        print_section("RESPONSE")
        print(colored("-" * 50, Colors.DIM))
        print(result.response)
        print(colored("-" * 50, Colors.DIM))

        return {
            'success': result.success,
            'time_ms': result.execution_time_ms,
            'agents': result.agents_used,
            'intent': result.query_intent,
            'response': result.response
        }


def print_help():
    """Print help information"""
    print_header("CLI TEST TOOL - HELP")
    print("""
Commands:
  /quit, /exit    Exit the CLI
  /agents         List all registered agents with details
  /stats          Show session statistics
  /clear          Clear the screen
  /verbose on/off Toggle verbose logging
  /help           Show this help message

Example queries:
  - Wie viele Bagger haben wir?
  - Liste alle Kettenbagger von Liebherr
  - Welche Geraete haben GPS?
  - Empfehle mir einen Bagger fuer schwere Einsaetze
  - Vergleich Kettenbagger vs Mobilbagger
""")


def print_stats(stats: SessionStats):
    """Print session statistics"""
    print_header("SESSION STATISTICS")
    print(f"""
  Session Duration: {stats.session_duration()}

  Queries:
    Total:    {stats.queries_total}
    Success:  {colored(str(stats.queries_success), Colors.GREEN)}
    Failed:   {colored(str(stats.queries_failed), Colors.RED)}

  Performance:
    Total Time:   {stats.total_time_ms}ms
    Average Time: {stats.avg_time_ms():.0f}ms

  Agent Usage:""")

    for agent, count in sorted(stats.agents_invoked.items(), key=lambda x: -x[1]):
        bar = colored("█" * min(count * 2, 20), Colors.CYAN)
        print(f"    {agent:15} {bar} {count}")


def clear_screen():
    """Clear the terminal screen"""
    os.system('cls' if os.name == 'nt' else 'clear')


async def main():
    """Main CLI loop"""
    clear_screen()

    print(colored("""
    ╔═══════════════════════════════════════════════════════════════╗
    ║                                                               ║
    ║   RUKO Multi-Agent System - Interactive CLI Tester            ║
    ║                                                               ║
    ║   Type your questions to test the system                      ║
    ║   Type /help for available commands                           ║
    ║   Type /quit to exit                                          ║
    ║                                                               ║
    ╚═══════════════════════════════════════════════════════════════╝
    """, Colors.CYAN))

    # Initialize the system
    try:
        agent_system = VerboseAgentSystem()
    except Exception as e:
        print_error(f"Failed to initialize agent system: {e}")
        return

    stats = SessionStats()

    print(colored("\nReady for queries!", Colors.GREEN + Colors.BOLD))

    while True:
        try:
            # Get user input
            print()
            user_input = input(colored("You > ", Colors.BOLD + Colors.BLUE)).strip()

            if not user_input:
                continue

            # Handle commands
            if user_input.startswith('/'):
                cmd = user_input.lower().split()[0]
                args = user_input.split()[1:] if len(user_input.split()) > 1 else []

                if cmd in ['/quit', '/exit']:
                    print_stats(stats)
                    print(colored("\nGoodbye!", Colors.CYAN))
                    break

                elif cmd == '/help':
                    print_help()

                elif cmd == '/agents':
                    agent_system._print_registered_agents()

                elif cmd == '/stats':
                    print_stats(stats)

                elif cmd == '/clear':
                    clear_screen()

                elif cmd == '/verbose':
                    if args and args[0] in ['on', 'off']:
                        agent_system.system.config.verbose = (args[0] == 'on')
                        print_info(f"Verbose mode: {args[0]}")
                    else:
                        current = "on" if agent_system.system.config.verbose else "off"
                        print_info(f"Verbose mode is currently: {current}")
                        print_info("Usage: /verbose on|off")

                else:
                    print_error(f"Unknown command: {cmd}")
                    print_info("Type /help for available commands")

            else:
                # Process as a query
                try:
                    result = await agent_system.process_query(user_input)
                    stats.add_query(
                        success=result['success'],
                        time_ms=result['time_ms'],
                        agents=result['agents']
                    )
                except Exception as e:
                    print_error(f"Query processing failed: {e}")
                    stats.add_query(success=False, time_ms=0, agents=[])

        except KeyboardInterrupt:
            print(colored("\n\nInterrupted by user", Colors.YELLOW))
            print_stats(stats)
            print(colored("\nGoodbye!", Colors.CYAN))
            break

        except EOFError:
            print(colored("\n\nEnd of input", Colors.YELLOW))
            break


if __name__ == "__main__":
    asyncio.run(main())
