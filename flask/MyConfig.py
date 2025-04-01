class MyConfig:
    def __init__(self):
        self.web_host = "127.0.0.1"
        self.web_port = 8001

        # OSS配置项
        self.endpoint = 'https://oss-cn-beijing.aliyuncs.com'
        self.AccessKey_ID = "LTAI5tGqeABdr3y8Cz2DnaT8"
        self.AccessKey_Secret="ytmA19k8bO58GvDR2F2g9qtqYydqpz"
        self.bucket_name='cmvclab'
        
        # 待检测文件路径
        self.universalImageFile="/root/FakeImageDetection-master/FakeImageDetection-master/testImages/1_fake"
        self.localMaskImageFile="/root/IML-ViT-main/IML-ViT-main/imagesFromFE/Tp"
        self.localFaceImageFile="/root/HDIL/imagesFromFE"
        
        self.maskSvgFile="/root/IML-ViT-main/IML-ViT-main/maskResults/mask.svg"
        # spring服务器地址
        self.springServerIp = "http://112.124.32.48:8080/flask"
        # 10.147.18.87
        self.attempts = 3
        self.testSpringServerIp="http://112.124.32.48:8081/flask"
    # 将经常用到的参数都放在配置里面
        