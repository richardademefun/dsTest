"""Simple device type processor"""


class DeviceTypeScore:

    def __init__(self, weights):
        self.weights = weights

    def run(self, input_data):
        data = input_data.copy()
        return data['device_type'].map(self.weights)
