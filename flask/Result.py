from flask import jsonify

class Result:
    def __init__(self, code=0, msg=None, data=None):
        self.returnDict = {
            "code": code,
            "msg": msg,
            "data": data
        }

    @classmethod
    def success(cls, returnData, msg="成功"):
        instance = cls(code=200, msg=msg, data=returnData)
        return instance.returnDict

    @classmethod
    def error(cls, msg="失败"):
        instance = cls(code=400,msg=msg)
        return instance.returnDict

if __name__ == "__main__":
    print(Result.success({"id":0, "df":"dcfadc"}))