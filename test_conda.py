import  torch
if __name__ == '__main__':

   print(torch.__version__)  # 输出版本，如2.3.0
   print(torch.cuda.is_available())  # 应输出True
   print(torch.backends.cudnn.is_available())  # 输出 True 表示可用
