import argparse
import ast
import datetime
import glob
import http.server
import json
import os
import subprocess
import urllib.parse
import urllib.request
import webbrowser
from pathlib import Path

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


class ProxyHandler(http.server.SimpleHTTPRequestHandler):
    backend = 'http://localhost:11434/v1'

    def log_message(self, format, *args):
        # Silence routine /tasks polling so the console doesn't get spammed.
        if self.path and '/tasks' in self.path:
            return
        super().log_message(format, *args)

    def _send_json(self, status, data):
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', len(body))
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.startswith('/v1'):
            self.proxy()
        elif self.path.startswith('/tool/'):
            self.handle_tool()
        elif self.path.startswith('/tasks'):
            self.handle_tasks()
        else:
            super().do_GET()

    def do_POST(self):
        if self.path.startswith('/v1'):
            self.proxy()
        elif self.path.startswith('/tool/'):
            self.handle_tool()
        elif self.path.startswith('/tasks'):
            self.handle_tasks()
        else:
            self.send_error(405)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, DELETE, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', '*')
        self.end_headers()

    def _safe_path(self, rel):
        if not rel:
            rel = ''
        # Use pathlib for cross-platform path normalization
        from pathlib import PurePosixPath
        rel = rel.strip('/')
        parts = PurePosixPath(rel.replace('\\', '/')).parts
        safe = []
        for p in parts:
            if p == '..' or p.startswith('..'):
                return None
            if p:
                safe.append(p)
        if not safe:
            return PROJECT_ROOT
        return os.path.abspath(os.path.join(PROJECT_ROOT, *safe))

    def handle_tool(self):
        parsed = urllib.parse.urlparse(self.path)
        name = parsed.path[len('/tool/'):].strip('/')
        params = urllib.parse.parse_qs(parsed.query)

        if name == 'get_current_time':
            self._send_json(200, {'result': datetime.datetime.now().isoformat()})
            return

        if name == 'calculate':
            length = int(self.headers.get('Content-Length', 0))
            raw = self.rfile.read(length).decode() if length else '{}'
            try:
                payload = json.loads(raw) if raw else {}
            except Exception:
                payload = {}
            expr = payload.get('expression', '')
            try:
                result = self._safe_eval(expr)
                self._send_json(200, {'result': result})
            except Exception as e:
                self._send_json(400, {'error': str(e)})
            return

        if name == 'read_file':
            rel = params.get('path', [''])[0]
            path = self._safe_path(rel)
            if not path or not path.startswith(PROJECT_ROOT):
                self._send_json(403, {'error': 'Forbidden'})
                return
            if not os.path.isfile(path):
                self._send_json(404, {'error': 'Not found'})
                return
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    self._send_json(200, {'result': f.read(), 'path': rel})
            except Exception as e:
                self._send_json(500, {'error': str(e)})
            return

        if name == 'list_directory':
            rel = params.get('path', [''])[0]
            path = self._safe_path(rel)
            if not path or not path.startswith(PROJECT_ROOT):
                self._send_json(403, {'error': 'Forbidden'})
                return
            if not os.path.isdir(path):
                self._send_json(404, {'error': 'Not found'})
                return
            try:
                entries = []
                for n in sorted(os.listdir(path)):
                    full = os.path.join(path, n)
                    st = os.stat(full)
                    entries.append({'name': n, 'type': 'directory' if os.path.isdir(full) else 'file', 'size': st.st_size})
                self._send_json(200, {'result': entries})
            except Exception as e:
                self._send_json(500, {'error': str(e)})
            return

        if name == 'write_file':
            if self.command != 'POST':
                self._send_json(405, {'error': 'Method not allowed'})
                return
            length = int(self.headers.get('Content-Length', 0))
            raw = self.rfile.read(length).decode() if length else '{}'
            try:
                payload = json.loads(raw) if raw else {}
            except Exception:
                payload = {}
            rel = payload.get('path', '')
            content = payload.get('content', '')
            path = self._safe_path(rel)
            if not path or not path.startswith(PROJECT_ROOT):
                self._send_json(403, {'error': 'Forbidden'})
                return
            try:
                os.makedirs(os.path.dirname(path), exist_ok=True)
                with open(path, 'w', encoding='utf-8') as f:
                    f.write(content)
                self._send_json(200, {'result': 'saved', 'path': rel, 'bytes': len(content.encode('utf-8'))})
            except Exception as e:
                self._send_json(500, {'error': str(e)})
            return

        if name == 'search_files':
            search_query = ''
            if self.command == 'POST':
                length = int(self.headers.get('Content-Length', 0))
                raw = self.rfile.read(length).decode() if length else '{}'
                try:
                    payload = json.loads(raw) if raw else {}
                except Exception:
                    payload = {}
                search_query = payload.get('query', '')
                rel = payload.get('path', '')
            else:
                search_query = params.get('query', [''])[0]
                rel = params.get('path', [''])[0]
            if not search_query:
                self._send_json(400, {'error': 'Missing query'})
                return
            path = self._safe_path(rel)
            if not path or not path.startswith(PROJECT_ROOT):
                self._send_json(403, {'error': 'Forbidden'})
                return
            if not os.path.isdir(path):
                self._send_json(404, {'error': 'Not found'})
                return
            results = []
            term = search_query.lower()
            try:
                for root, dirs, files in os.walk(path):
                    dirs[:] = [d for d in dirs if d not in ('__pycache__', '.git', 'node_modules')]
                    for filename in files:
                        full = os.path.join(root, filename)
                        rel_full = os.path.relpath(full, PROJECT_ROOT).replace('\\', '/')
                        if term in filename.lower() or term in rel_full.lower():
                            results.append({'path': rel_full, 'type': 'filename'})
                            continue
                        try:
                            with open(full, 'r', encoding='utf-8', errors='ignore') as f:
                                text = f.read(200000)
                            if term in text.lower():
                                idx = text.lower().find(term)
                                snippet = text[max(0, idx - 60):idx + 100]
                                results.append({'path': rel_full, 'type': 'content', 'snippet': snippet})
                        except Exception:
                            pass
                        if len(results) >= 50:
                            break
                    if len(results) >= 50:
                        break
                self._send_json(200, {'result': results})
            except Exception as e:
                self._send_json(500, {'error': str(e)})
            return

        self._send_json(404, {'error': 'Unknown tool'})

    def _tasks_dir(self):
        return os.path.join(PROJECT_ROOT, 'research', 'tasks')

    def _read_status(self, task_id):
        path = os.path.join(self._tasks_dir(), task_id, 'status.json')
        if os.path.isfile(path):
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception:
                pass
        return None

    def _read_log_tail(self, task_id, lines=100):
        path = os.path.join(self._tasks_dir(), task_id, 'log.txt')
        if not os.path.isfile(path):
            return []
        try:
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                all_lines = f.readlines()
            return [l.rstrip('\n') for l in all_lines[-lines:]]
        except Exception:
            return []

    def handle_tasks(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        parts = [p for p in path.split('/') if p]

        if self.command == 'GET' and len(parts) == 1 and parts[0] == 'tasks':
            tasks = []
            tasks_dir = self._tasks_dir()
            if os.path.isdir(tasks_dir):
                for status_path in glob.glob(os.path.join(tasks_dir, '*', 'status.json')):
                    try:
                        with open(status_path, 'r', encoding='utf-8') as f:
                            status = json.load(f)
                        tasks.append(status)
                    except Exception:
                        pass
            tasks.sort(key=lambda t: t.get('updated_at', ''), reverse=True)
            self._send_json(200, {'tasks': tasks})
            return

        if self.command == 'GET' and len(parts) == 2 and parts[0] == 'tasks':
            task_id = parts[1]
            status = self._read_status(task_id)
            if not status:
                self._send_json(404, {'error': 'Task not found'})
                return
            status['log_tail'] = self._read_log_tail(task_id)
            self._send_json(200, status)
            return

        if self.command == 'POST' and len(parts) == 3 and parts[0] == 'tasks' and parts[2] == 'stop':
            task_id = parts[1]
            status = self._read_status(task_id)
            if not status:
                self._send_json(404, {'error': 'Task not found'})
                return
            pid = status.get('pid')
            try:
                subprocess.run(['taskkill', '/PID', str(pid), '/F'], check=False, capture_output=True)
                status['status'] = 'stopped'
                self._send_json(200, status)
            except Exception as e:
                self._send_json(500, {'error': str(e)})
            return

        self._send_json(404, {'error': 'Unknown tasks endpoint'})

    def _safe_eval(self, expression):
        if not expression:
            raise ValueError('Empty expression')
        tree = ast.parse(expression, mode='eval')
        allowed = (ast.Expression, ast.BinOp, ast.UnaryOp, ast.Constant, ast.Num, ast.Load,
                   ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.Mod, ast.USub, ast.UAdd)
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and not isinstance(node.value, (int, float, bool)):
                raise ValueError('Unsupported constant')
            if not isinstance(node, allowed):
                raise ValueError('Unsupported expression')
        return eval(compile(tree, '<string>', 'eval'), {'__builtins__': {}})

    def proxy(self):
        target = self.path
        if target.startswith('/v1'):
            target = target[3:]
        if not target.startswith('/'):
            target = '/' + target
        url = self.backend.rstrip('/') + target
        content_length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(content_length) if content_length else None
        req_headers = {}
        for h in ('Content-Type', 'Authorization', 'Accept'):
            if h in self.headers:
                req_headers[h] = self.headers[h]
        try:
            import sys
            print('PROXY URL', url, file=sys.stderr)
            req = urllib.request.Request(url, data=body, method=self.command, headers=req_headers)
            with urllib.request.urlopen(req) as resp:
                self.send_response(resp.status)
                for k, v in resp.headers.items():
                    if k.lower() in ('transfer-encoding', 'content-length', 'connection'):
                        continue
                    self.send_header(k, v)
                self.send_header('Connection', 'close')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                while True:
                    data = resp.read(8192)
                    if not data:
                        break
                    self.wfile.write(data)
                    self.wfile.flush()
        except urllib.error.HTTPError as e:
            self.send_response(e.code)
            self.end_headers()
            self.wfile.write(e.read())
        except Exception as e:
            import sys
            print('PROXY ERROR', url, e, file=sys.stderr)
            self.send_response(502)
            self.end_headers()
            self.wfile.write(str(e).encode())


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8080)
    parser.add_argument('--backend', default='http://localhost:11434/v1')
    parser.add_argument('--no-browser', action='store_true')
    args = parser.parse_args()
    ProxyHandler.backend = args.backend
    server = http.server.ThreadingHTTPServer(('0.0.0.0', args.port), ProxyHandler)
    print(f'Serving on http://localhost:{args.port}')
    print(f'Proxying /v1/* to {args.backend}')
    if not args.no_browser:
        webbrowser.open(f'http://localhost:{args.port}')
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    from research.dx_setup import setup
    setup()
    run()
