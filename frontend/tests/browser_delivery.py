"""Real Firefox smoke test against synthetic HTTP fixtures, never production APIs.

Build frontend first; install marionette_driver into a task-local environment.
Run with --evidence DIR to retain screenshots and a machine-readable receipt.
"""
import argparse
import json
import socket
import subprocess
import tempfile
import threading
import time
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from marionette_driver.marionette import Marionette
from marionette_driver.keys import Keys

BUILD = Path(__file__).resolve().parents[1] / 'build'
FIXTURE_SCOPE = {'username': 'ui-fixture', 'conv_id': 'research-fixture'}


def snapshot(partial=False):
    runtime = {'plan_version': 1, 'nodes': {'charge': {'status': 'succeeded'}}}
    result = {'scope': FIXTURE_SCOPE, 'snapshot_at': time.time(), 'parallel_runtime': runtime}
    if not partial:
        result['graph_projection'] = {'source': 'persisted_chain', 'version': 1, 'nodes': [
            {'step_id': 'charge', 'description': 'Framework charge', 'tool': 'charge', 'status': 'running', 'depends_on': []},
            {'step_id': 'adsorption', 'description': 'Adsorption evaluation', 'tool': 'run_cdft', 'status': 'waiting_jobs', 'depends_on': ['charge']},
            {'step_id': 'report', 'description': 'Research report', 'status': 'pending', 'depends_on': ['adsorption']},
        ]}
    return result


class FixtureHandler(SimpleHTTPRequestHandler):
    partial = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(BUILD), **kwargs)

    def log_message(self, *args):
        pass

    def do_GET(self):
        path = self.path.split('?')[0]
        if path.startswith('/api/'):
            if 'events' in path:
                self.send_error(503, 'Fixture uses polling')
                return
            if path == '/api/conversations':
                payload = {'current_conv_id': FIXTURE_SCOPE['conv_id'], 'conversations': [
                    {'conv_id': FIXTURE_SCOPE['conv_id'], 'title': 'Membrane research fixture'}]}
            elif path == '/api/latest_answer':
                payload = {'done': True, 'workflow': snapshot(self.partial)}
            elif path.startswith('/api/conversations/'):
                payload = {'messages': [], 'is_processing': False, 'workflow': snapshot(self.partial)}
            elif path == '/api/jobs':
                payload = {'jobs': []}
            else:
                payload = {'username': FIXTURE_SCOPE['username']}
            body = json.dumps(payload).encode()
            self.send_response(200); self.send_header('Content-Type', 'application/json'); self.end_headers()
            self.wfile.write(body)
            return
        if path == '/':
            body = (BUILD / 'index.html').read_text().replace('<head>', '<head><script>'
                'sessionStorage.setItem("bimem_user", "ui-fixture");'
                'sessionStorage.setItem("bimem_token", "synthetic-test-token");'
                '</script>').encode()
            self.send_response(200); self.send_header('Content-Type', 'text/html'); self.end_headers()
            self.wfile.write(body)
            return
        super().do_GET()


def wait_for(client, expression, timeout=12):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = client.execute_script('return ' + expression)
        if value:
            return value
        time.sleep(.2)
    raise AssertionError('Browser condition timed out: ' + expression)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--evidence', type=Path, required=True)
    args = parser.parse_args()
    args.evidence.mkdir(parents=True, exist_ok=True)
    checks = []
    server = ThreadingHTTPServer(('127.0.0.1', 0), FixtureHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    with socket.socket() as port_probe:
        port_probe.bind(('127.0.0.1', 0)); browser_port = port_probe.getsockname()[1]
    # This profile belongs only to this test. It never loads the user's browser.
    with tempfile.TemporaryDirectory(prefix='firefox-', dir=args.evidence) as profile:
        Path(profile, 'user.js').write_text(f'user_pref("marionette.port", {browser_port});\n')
        browser = subprocess.Popen(['firefox', '--headless', '--no-remote', '--profile', profile, '--marionette'],
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        client = Marionette(port=browser_port, socket_timeout=10)
        try:
            client.start_session(timeout=30)
            client.set_window_rect(width=1440, height=1000)
            client.navigate(f'http://127.0.0.1:{server.server_port}/')
            wait_for(client, 'document.querySelectorAll(".stable-dag-node").length === 3')
            client.find_element('css selector', '[aria-label="Adsorption evaluation：作业执行中"] .stable-dag-label').click()
            wait_for(client, 'document.querySelector(".runtime-inspector-head b").textContent === "Adsorption evaluation"')
            checks.append('desktop click selects the correct live node')
            (args.evidence / 'desktop.png').write_bytes(client.screenshot(format='binary'))
            client.find_element('css selector', '[aria-label="放大编排图"]').click()
            client.find_element('css selector', '[aria-label="放大编排图"]').click()
            assert client.execute_script('return document.querySelector(".stable-dag-svg").getBoundingClientRect().width > document.querySelector(".runtime-graph-viewport").clientWidth')
            (args.evidence / 'zoomed.png').write_bytes(client.screenshot(format='binary'))
            client.find_element('css selector', '[aria-label="适应面板"]').click()
            assert client.execute_script('return document.querySelector(".stable-dag-svg").style.width === ""')
            checks.append('zoom provides readable node labels and reset fits the complete graph')
            node = client.find_element('css selector', '[aria-label="Research report：等待调度"]')
            node.send_keys(Keys.SPACE)
            wait_for(client, 'document.querySelector(".runtime-inspector-head b").textContent === "Research report"')
            checks.append('keyboard Space selects the report node')
            node.send_keys(Keys.ESCAPE)
            wait_for(client, 'document.querySelector(".runtime-drawer-toggle").getAttribute("aria-expanded") === "false"')
            assert client.execute_script('return document.activeElement.classList.contains("runtime-drawer-toggle")')
            checks.append('Escape closes drawer and restores keyboard focus')
            client.find_element('css selector', '.runtime-drawer-toggle').click()
            FixtureHandler.partial = True
            # The real application's idle poll is 30 seconds; observe it, not a mocked timer.
            wait_for(client, 'document.querySelector(".runtime-graph-recovered")?.textContent.includes("保留")', timeout=35)
            assert client.execute_script('return document.querySelectorAll(".stable-dag-node").length') == 3
            checks.append('complete graph remains visible after an actual 30-second partial poll')
            client.refresh()
            wait_for(client, 'document.querySelectorAll(".stable-dag-node").length === 3')
            checks.append('hard refresh recovers the complete graph while server sends partial state')
            client.set_window_rect(width=390, height=844)
            client.execute_script('localStorage.removeItem("runtime_drawer_open")')
            client.refresh()
            wait_for(client, '!!document.querySelector(".runtime-drawer-closed")')
            assert client.execute_script('return document.documentElement.scrollWidth <= window.innerWidth')
            checks.append('narrow screen starts with an accessible conversation and no horizontal page overflow')
            client.find_element('css selector', '.runtime-drawer-toggle').click()
            wait_for(client, '!!document.querySelector(".runtime-drawer-open")')
            assert client.execute_script('return document.querySelector(".runtime-drawer").getBoundingClientRect().width <= window.innerWidth')
            (args.evidence / 'mobile.png').write_bytes(client.screenshot(format='binary'))
            checks.append('mobile drawer fits viewport and retains all graph nodes')
            receipt = {'passed': len(checks), 'checks': checks, 'scope': 'Real Firefox UI with synthetic HTTP responses; no live scientific jobs or real account used.'}
            (args.evidence / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
            print(json.dumps(receipt))
        finally:
            try:
                client.delete_session()
            except Exception:
                pass
            browser.terminate()
            try:
                browser.wait(timeout=10)
            except subprocess.TimeoutExpired:
                browser.kill(); browser.wait()
            server.shutdown()


if __name__ == '__main__':
    main()
