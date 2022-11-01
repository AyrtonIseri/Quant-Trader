from numpy import take
from utils.update_data import Updater
from utils.load import Loader
from agents.Agent import Agent
from models.Indicators.Signals.SimpleMovingAverage import SimpleMovingAverage

Updater().update_data()
historical_data = Loader().load_data()

model1_parameters = {
    'take_profit': 0.15,
    'stop_loss': -0.10,
    'indicator': SimpleMovingAverage(fast_curve = 5, slow_curve = 30),
    'invest_percentage': 0.2
}
model1_weight = 0.5

model2_parameters = {
    'take_profit': 0.15,
    'stop_loss': -0.10,
    'indicator': SimpleMovingAverage(fast_curve = 5, slow_curve = 30),
    'invest_percentage': 0.25
}
model2_weight = 0.5

models = [model1_parameters, model2_parameters]
models_weights = [model1_weight, model2_weight]

account_balance = 10000

agent = Agent(account_balance = account_balance, models_weights = models_weights,
              models_kwargs = models, historical_data = historical_data)

agent.run()

for model in agent._models:
    print(model.get_account_balance())

