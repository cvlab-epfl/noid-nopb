from model.model import MotionModel

def get_model(model_conf, data_conf):
    return MotionModel(model_conf, data_conf)