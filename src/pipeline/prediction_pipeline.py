import sys
import os

from src.logger import logging
from src.exception import CustomException




import pandas as pd

from src.utils import load_object

class predictPipeline:
    def __init__(self):
        pass
    def predict(self, features):
        try:
            ##load both pkl files
            preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')
            model_path = os.path.join('artifacts', 'model.pkl')

            preprocessor = load_object(preprocessor_path)
            model = load_object(model_path)

            data_scaled = preprocessor.transform(features)
            pred = model.predict(data_scaled)
            return pred



        except Exception as e:
            logging.info("Exception occured in prediction")
            raise CustomException(e,sys)
        
class CustomData:
    def __init__(self,
                 Delivery_person_Age:float,
                 Delivery_person_Ratings: float,
                 Weather_conditions:str,
                 Road_traffic_density:str,
                 Vehicle_condition: float,
                 Type_of_order:str,
                 Type_of_vehicle:str,
                 multiple_deliveries:float,
                 Festival:str,
                 City:str,
                 distance:float,
                 Day:int,
                 Month:int,
                 Year:int):
        self.Delivery_person_Age = Delivery_person_Age
        self.Delivery_person_Ratings = Delivery_person_Ratings
        self.Weather_conditions = Weather_conditions
        self.Road_traffic_density=Road_traffic_density
        self.Vehicle_condition = Vehicle_condition
        self.Type_of_order=Type_of_order
        self.Type_of_vehicle=Type_of_vehicle
        self.multiple_deliveries= multiple_deliveries
        self.Festival=Festival
        self.City = City
        self.distance=distance
        self.Day = Day
        self.Month = Month
        self.Year = Year

    def get_data_as_dataframe(self):
        try:
            custome_data_input_dict = {
                'Delivery_person_Age':[self.Delivery_person_Age],
                'Delivery_person_Ratings':[self.Delivery_person_Ratings],
                'Weather_conditions':[self.Weather_conditions],
                'Road_traffic_density':[self.Road_traffic_density],
                'Vehicle_condition':[self.Vehicle_condition],
                'Type_of_order': [self.Type_of_order],
                'Type_of_vehicle': [self.Type_of_vehicle],
                'multiple_deliveries':[self.multiple_deliveries],
                'Festival':[self.Festival],
                'City':[self.City],
                'distance':[self.distance],
                'Day':[self.Day],
                'Month':[self.Month],
                'Year':[self.Year]
            }

            df = pd.DataFrame(custome_data_input_dict)
            logging.info('Data InformationGathered')
            return df
        except Exception as e:
            logging.info('Exception occured in ')
            raise CustomException(e,sys)


        
    