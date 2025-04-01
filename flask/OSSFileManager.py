# -*- coding: utf-8 -*-
import oss2
from itertools import islice
import os
from datetime import datetime
import snowflake.client

class OSSFileManager():
    def __init__(self, AccessKey_ID, AccessKey_Secret, endpoint, bucket_name) -> None:
        auth = oss2.Auth(AccessKey_ID, AccessKey_Secret)
        self.bucket = oss2.Bucket(auth, endpoint, bucket_name)
    
    # 获取文件列表
    # def getFileList(self, prefix=None):
    #     for b in oss2.ObjectIterator(self.bucket, prefix=prefix):
    #         print(b.key)

    # 上传文件        
    # def uploadFile(self, upFileName:str, LocalFile:str):
    #     fileType = upFileName.split(".")[-1]
    #     if fileType == "mp4":
    #         upFileName = self.videoPath + upFileName
    #     elif fileType == "jpg":
    #         upFileName = self.imagePath + upFileName
    #     self.bucket.put_object_from_file(upFileName, LocalFile)
    #     os.remove(LocalFile)
    #     return upFileName
    
    # 下载文件
    # def getFile(self,fromPath:str,toPath:str):
    #     self.bucket.get_object_to_file(fromPath,toPath)



if __name__ == "__main__":
    df = OSSFileManager("LTAI5tGqeABdr3y8Cz2DnaT8", "ytmA19k8bO58GvDR2F2g9qtqYydqpz", 'https://oss-cn-beijing.aliyuncs.com', 'cmvclab', "text2video/", None)
    # df.uploadFile("4870683326539632641.mp4","/root/autodl-tmp/generationVideo/4870683326539632641.mp4" )
    df.getFileList()


