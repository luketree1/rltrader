import numpy as np
from quantylab.rltrader import utils


class Agent:
    # 에이전트 상태가 구성하는 값 개수
    # 주식 보유 비율, 현재 손익, 평균 매수 단가 대비 등락률
    STATE_DIM = 4

    # 매매 수수료 및 세금
    TRADING_CHARGE = 0.0000 # 거래 수수료 0.015%
    # TRADING_CHARGE = 0.00011  # 거래 수수료 0.011%
    # TRADING_CHARGE = 0  # 거래 수수료 미적용
    # TRADING_TAX = 0.0025  # 거래세 0.25%
    TRADING_TAX = 0  # 거래세 미적용

    # 행동
    ACTION_BUY = 0  # 매수
    ACTION_SELL = 1  # 매도
    ACTION_HOLD = 2  # 관망
    # 인공 신경망에서 확률을 구할 행동들
    ACTIONS = [ACTION_BUY, ACTION_SELL, ACTION_HOLD]
    NUM_ACTIONS = len(ACTIONS)  # 인공 신경망에서 고려할 출력값의 개수

    def __init__(self, environment, initial_balance, min_trading_price, max_trading_price):
        # 현재 주식 가격을 가져오기 위해 환경 참조
        self.environment = environment
        self.initial_balance = initial_balance  # 초기 자본금

        # 최소 단일 매매 금액, 최대 단일 매매 금액
        self.min_trading_price = min_trading_price
        self.max_trading_price = max_trading_price

        # Agent 클래스의 속성
        self.balance = initial_balance  # 현재 현금 잔고
        self.position = 0  # 보유 포지션 수
        self.num_stocks = 0  # 보유 주식 수
        self.total_long_position = 0
        self.total_long_position_cost = 0
        self.total_short_position = 0
        self.total_short_position_cost = 0

        # 포트폴리오 가치: balance + num_stocks * {현재 주식 가격}
        # 포트폴리오 가치: balance + unrealized_profit -> (position 수 * (cur price - entry price)
        self.portfolio_value = 0
        self.num_buy = 0  # 매수 횟수
        self.num_sell = 0  # 매도 횟수
        self.num_hold = 0  # 관망 횟수

        # Agent 클래스의 상태
        self.ratio_hold = 0  # 주식 보유 비율
        self.profitloss = 0  # 현재 손익
        self.avg_buy_price = 0  # 주당 매수 단가

    def reset(self):
        self.balance = self.initial_balance
        self.position = 0  # 보유 포지션 수
        self.num_stocks = 0
        self.total_long_position = 0
        self.total_long_position_cost = 0
        self.total_short_position = 0
        self.total_short_position_cost = 0
        self.portfolio_value = self.initial_balance
        self.num_buy = 0
        self.num_sell = 0
        self.num_hold = 0
        self.ratio_hold = 0
        self.profitloss = 0
        self.avg_buy_price = 0

    def set_balance(self, balance):
        self.initial_balance = balance

    def get_states(self):
        self.ratio_hold = self.num_stocks * self.environment.get_price() \
                          / self.portfolio_value
        return (
            self.ratio_hold,
            self.profitloss,
            self.num_buy,
            self.num_hold,
            # self.num_stocks,
            # (self.environment.get_price() / self.avg_buy_price) - 1 \
                # if self.avg_buy_price > 0 else 0
        )

    def decide_action(self, pred_value, pred_policy, epsilon):
        confidence = 0.

        pred = pred_policy
        if pred is None:
            pred = pred_value

        if pred is None:
            # 예측 값이 없을 경우 탐험
            epsilon = 1
        else:
            # 값이 모두 같은 경우 탐험
            maxpred = np.max(pred)
            if (pred == maxpred).all():
                epsilon = 1

            # if pred_policy is not None:
            #     if np.max(pred_policy) - np.min(pred_policy) < 0.05:
            #         epsilon = 1

        # 탐험 결정
        if np.random.rand() < epsilon:
            exploration = True
            action = np.random.randint(self.NUM_ACTIONS)
        else:
            exploration = False
            action = np.argmax(pred)

        confidence = .5
        if pred_policy is not None:
            confidence = pred[action]
        elif pred_value is not None:
            confidence = utils.sigmoid(pred[action])

        return action, confidence, exploration

    def validate_action(self, action):

        if self.num_stocks * self.environment.get_price() > self.balance:
            return False
        # if action == Agent.ACTION_BUY:
        #     # 적어도 1주를 살 수 있는지 확인
        #     if self.balance < 1000:
        #         return False
        # elif action == Agent.ACTION_SELL:
        #     # 주식 잔고가 있는지 확인
        #     if self.num_stocks <= 0:
        #         return False
        return True

    # balance = max position value

    def decide_trading_unit(self, confidence):
        if confidence < 0:
            confidence = 0
        if confidence > 1:
            confidence = 1
        return (self.min_trading_price + confidence * (self.max_trading_price - self.min_trading_price)) \
            / self.environment.get_price()

    def act2(self, action, confidence):
        curr_price = self.environment.get_price()
        trading_unit = self.decide_trading_unit(confidence)

        if action == Agent.ACTION_BUY and self.position * curr_price > self.portfolio_value:
            action = Agent.ACTION_HOLD

        if action == Agent.ACTION_SELL and self.position * curr_price < -self.portfolio_value:
            action = Agent.ACTION_HOLD

        if action == Agent.ACTION_BUY:
            self.total_long_position += trading_unit
            self.total_long_position_cost += trading_unit * curr_price * (1 + self.TRADING_CHARGE)
            self.num_stocks += trading_unit
            self.position += trading_unit
            self.num_buy += 1
        elif action == Agent.ACTION_SELL:
            self.total_short_position += trading_unit
            self.total_short_position_cost += trading_unit * curr_price * (1 - self.TRADING_CHARGE)
            self.num_stocks -= trading_unit
            self.position -= trading_unit
            self.num_sell += 1
        elif action == Agent.ACTION_HOLD:
            self.num_hold += 1

        long_profit = 0
        if self.total_long_position != 0:
            long_profit = self.total_long_position*(curr_price - (self.total_long_position_cost / self.total_long_position))

        short_profit = 0
        if self.total_short_position != 0:
            short_profit = -1 * self.total_short_position*(curr_price - (self.total_short_position_cost / self.total_short_position))

        hold_reward = 0
        if action == Agent.ACTION_HOLD:
            hold_reward = 0.0001
        self.portfolio_value = self.balance + short_profit + long_profit
        self.profitloss = self.portfolio_value / self.initial_balance - 1
        return self.profitloss

    def act(self, action, confidence):
        if not self.validate_action(action):
            action = Agent.ACTION_HOLD

        # 환경에서 현재 가격 얻기
        curr_price = self.environment.get_price()

        # 매수
        if action == Agent.ACTION_BUY:

            # 매수할 단위를 판단
            trading_unit = self.decide_trading_unit(confidence)
            # print("ACTION_BUY",trading_unit)
            balance = (
                    self.balance - curr_price *
                    (1 + self.TRADING_CHARGE) * trading_unit
            )
            # 보유 현금이 모자랄 경우 보유 현금으로 가능한 만큼 최대한 매수
            if balance < 0:
                trading_unit = min(
                    int(self.balance / (curr_price * (1 + self.TRADING_CHARGE))),
                    int(self.max_trading_price / curr_price)
                )
            # 수수료를 적용하여 총 매수 금액 산정
            invest_amount = curr_price * (1 + self.TRADING_CHARGE) * trading_unit

            if invest_amount > 0:
                self.avg_buy_price = \
                    (self.avg_buy_price * self.num_stocks + curr_price * trading_unit) \
                    / (self.num_stocks + trading_unit)  # 주당 매수 단가 갱신
                self.balance -= invest_amount  # 보유 현금을 갱신
                self.num_stocks += trading_unit  # 보유 주식 수를 갱신
                self.num_buy += 1  # 매수 횟수 증가


        # 매도
        elif action == Agent.ACTION_SELL:
            # print("ACTION_SELL")

            # 매도할 단위를 판단
            trading_unit = self.decide_trading_unit(confidence)
            # print("ACTION_SELL",trading_unit)
            # 보유 주식이 모자랄 경우 가능한 만큼 최대한 매도
            trading_unit = min(trading_unit, self.num_stocks)
            # 매도
            invest_amount = curr_price * (
                    1 - (self.TRADING_TAX + self.TRADING_CHARGE)) * trading_unit
            if invest_amount > 0:
                # 주당 매수 단가 갱신
                self.avg_buy_price = \
                    (self.avg_buy_price * self.num_stocks - curr_price * trading_unit) \
                    / (self.num_stocks - trading_unit) \
                        if self.num_stocks > trading_unit else 0
                self.num_stocks -= trading_unit  # 보유 주식 수를 갱신
                self.balance += invest_amount  # 보유 현금을 갱신
                self.num_sell += 1  # 매도 횟수 증가

        # 관망
        elif action == Agent.ACTION_HOLD:
            self.num_hold += 1  # 관망 횟수 증가

        # 포트폴리오 가치 갱신
        self.portfolio_value = self.balance + curr_price * self.num_stocks
        self.profitloss = self.portfolio_value / self.initial_balance - 1
        return self.profitloss

#%%
