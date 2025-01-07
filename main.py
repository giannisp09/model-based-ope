'''
 this is the main file of the project where the model based  off policy evaluation is implemented
 '''

 from model_based import ModelBased


 # load a offline reinforcement learning dataset and train a model based off policy evaluation

    if __name__ == '__main__':
        # Define the hyperparameters.
        state_dim = 3
        action_dim = 1
        learning_rate = 1e-3
        weight_decay = 1e-4
        num_epochs = 100
        num_rollouts = 100
        horizon = 10

        # Initialize the model-based RL agent.
        model_based = ModelBased(state_dim, action_dim, learning_rate, weight_decay)

        # Train the model-based RL agent.
        model_based.train(num_epochs, num_rollouts, horizon)

        # Evaluate the model-based RL agent.
        model_based.evaluate(num_rollouts, horizon)
