# Stock Trading Market OpenAI Gym Environment with PARL

## Overview
base on :

## Reference

[1] https://github.com/wangshub/RL-Stock

[2] https://github.com/kh-kim/stock_market_reinforcement_learning

[3] https://github.com/PaddlePaddle/Paddle

[4] https://github.com/PaddlePaddle/PARL


## Requirements

- Python3.7
- Numpy
- parl
- paddle
- OpenAI Gym


GET DATA:

	$ python get_stock_data.py
	
run:

        $ python DDPG_STOCK.py
	
	$ python DQN_STOCK.py




######################################################################
######################################################################
#
# 7. 请选择你训练的最好的一次模型文件做评估
#
######################################################################
######################################################################
def stock_trade(stock_file,ckpt_files):
    ckpt = ckpt_files # 请设置ckpt为你训练中效果最好的一次评估保存的模型文件名称
    agent.restore(ckpt)
    # 创建环境
    df_test = pd.read_csv(stock_file)
    df_test = df_test.sort_values('date')
    # The algorithms require a vectorized environment to run
    env_test = StockTradingEnv(df_test)
    day_profits = []
    # The algorithms require a vectorized environment to run
    obs = env_test.reset()
    for i in range(len(df_test) - 1):
        batch_obs = np.expand_dims(obs, axis=0)
        action = agent.predict(batch_obs.astype('float32'))
        action = np.squeeze(action)

        # 给输出动作增加探索扰动，输出限制在 [-1.0, 1.0] 范围内
        action = np.random.normal(action, 1.0)
        action = np.clip(action, -1.0, 1.0)
        # 动作映射到对应的 实际动作取值范围 内, action_mapping是从parl.utils那里import进来的函数
        action = action_mapping(action, env.action_space.low[0],
                                env.action_space.high[0])
        next_obs, reward, done,info= env.step(action)
        Net_worth = env.render()
        day_profits.append(Net_worth)
    return day_profits
def test_a_stock_trade(stock_file,ckpt_files):
    daily_profits = stock_trade(stock_file,ckpt_files)
    daily_profits= pd.DataFrame(daily_profits)
    df_test = pd.read_csv(stock_file)
    df_test = df_test.sort_values('date')
    fig, ax = plt.subplots()
    x = df_test['date'].drop(0)
    y1 = daily_profits[0]
    ax.plot(x,y1)
    ax.grid()
    #ax.legend(prop=font)
    #plt.show()
    plt.savefig(f'test.png')
        
