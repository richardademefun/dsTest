"""Simple time of day processor"""
from dsTest.constants import hours_in_day


class TimeOfDayScore:

    def __init__(self, weights):
        self.weights = weights

    def run(self, input_data):
        data = input_data.copy()
        data['hour_of_week'] = data['weekday_user_tz'] * hours_in_day + data['hour_user_tz']
        data['hour_of_week_score'] = data['hour_of_week'].map(self.weights)

        return data['hour_of_week_score']
