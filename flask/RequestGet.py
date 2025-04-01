import multiprocessing
from flask import Flask, request
from flask_cors import CORS
from Result import Result
from MyConfig import MyConfig
from multiprocessing import Process, Queue
import sys
sys.path.append("/root/IML-ViT-main/IML-ViT-main")
sys.path.append('/root/HDIL/')
sys.path.append('/root/HDIL/getFaces/')
from ProcessWorker import universal_worker,localMask_worker,localFace_worker
import signal

class WebApplication:
    def __init__(self, app, universal_queue, localFace_queue, localMask_queue, config):
        self.universal_queue = universal_queue
        self.localFace_queue = localFace_queue
        self.localMask_queue = localMask_queue
        self.app = app
        self.config = config
        self.add_route()

    def add_route(self):
        @self.app.route("/model/detection", methods=["POST"])
        def detectionModelChose():
            server_request = request.json
            if "model" in server_request:
                model = server_request["model"]
                if model == "universal":
                    self.universal_queue.put(server_request)
                    print("已接收到任务：universal")
                elif model == "localFace":
                    self.localFace_queue.put(server_request)
                    print("已接收到任务：localFace")
                elif model == "localMask":
                    self.localMask_queue.put(server_request)
                    print("已接收到任务：localMask")
                    if server_request["type"]=="video":
                        return Result.success("localMask模型不可用于检测视频！")
                else:
                    return Result.error("模型选择参数错误")
                return Result.success(None)
            else:
                return Result.error("未指定模型")


def create_app():
    app = Flask(__name__)
    CORS(app)
    return app

def handle_sigint(signum, frame):
    print("接收到信号 SIGINT，正在停止...")
    sys.exit(0)

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)

    universal_queue = Queue()
    localFace_queue = Queue()
    localMask_queue = Queue()

    app = create_app()
    config = MyConfig()

    MyApp = WebApplication(app, universal_queue, localFace_queue, localMask_queue, config)

    worker_process_1 = Process(target=universal_worker, args=(MyApp.universal_queue,))
    worker_process_1.start()

    worker_process_2 = Process(target=localMask_worker, args=(MyApp.localMask_queue,))
    worker_process_2.start()
    
    worker_process_3 = Process(target=localFace_worker, args=(MyApp.localFace_queue,))
    worker_process_3.start()

    signal.signal(signal.SIGINT, handle_sigint)

    app.run(host=config.web_host, port=config.web_port, threaded=True, debug=False)
    # python /root/flask/RequestGet.py
