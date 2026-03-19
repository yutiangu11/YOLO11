# https://sso.openxlab.org.cn/usercenter?lang=zh-CN&tab=secret
import openxlab
openxlab.login(ak='2qogyplwv1d2y7an86bb', sk='oklyy3awbempa97zvmvqgrkyn084plqnznmwxvqj') # 进行登录，输入对应的AK/SK，可在个人中心添加AK/SK


from openxlab.dataset import info
info(dataset_repo='OpenDataLab/TinyPerson') #数据集信息查看
