## run Data Ingestion ##later mask it
'''
if __name__ == '__main__':
    obj = DataIngestion()
    train_data,test_data = obj.initiate_data_ingestion()
'''
## run Data Ingestion & DataTranformation ##later mask it
if __name__ == '__main__':
    obj = DataIngestion()
    train_data_path, test_data_path = obj.initiate_data_ingestion()
    data_tranformation = DataTransformation()
    train_arr, test_arr,_ = data_tranformation.initiate_data_transformation(train_data_path,test_data_path)
## Run training pipeline
if __name__=='__main__':
    obj=DataIngestion()
    train_data_path,test_data_path=obj.initiate_data_ingestion()
    data_transformation = DataTransformation()
    train_arr,test_arr,_=data_transformation.initiate_data_transformation(train_data_path,test_data_path)
    model_trainer=ModelTrainer()
    model_trainer.initate_model_training(train_arr,test_arr)