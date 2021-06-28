import math as m
import numpy as np
import random
from datetime import datetime
from gradient_free_optimizers import HillClimbingOptimizer, StochasticHillClimbingOptimizer

from snake_game import SnakeGame
from neural_network import NeuralNetwork
from helper import Helper

# Change OPTIMIZATION to True if you want to optimize hyperparams
OPTIMIZATION = False
LOAD_WEIGHTS = True


class Optimization:
    def __init__(self):
        pass

    def optimize(self, param):
        # Print parameters
        print('Parameters:', param)

        # Training data
        training_data = main(None)

        # Hyperparams retyping
        no_of_layers = int(param['no_of_layers'])
        no_of_neurons = param['no_of_neurons']
        lr = param['lr']
        batch_size = int(param['batch_size'])
        epochs = int(param['epochs'])

        # Initialize nn
        neural_network = NeuralNetwork(no_of_layers, no_of_neurons, lr)
        model = neural_network.model()
        # Train model
        model = self.train_model(training_data, model, batch_size, epochs)

        # Optimization process is based on this variable
        score = main(model)

        # Save logs
        self.save_logs(param, score)

        # Save weights
        neural_network.save_weights()

        # Return score
        return score

    def save_logs(self, param, score):
        with open('logs/scores_' + str(datetime.now().strftime("%Y%m%d%H%M%S")) + '.txt', 'a') as f:
            f.write(
                str('no_of_layers{}_no_of_neurons{}_snake_lr{}_batch_size{}_epochs{}_score{}'.format(
                    int(
                        param['no_of_layers']),
                    param['no_of_neurons'],
                    param['lr'],
                    param['batch_size'],
                    param['epochs'],
                    score)) + '\n')
            f.write('Params: ' + str(param) + '\n')

    def train_model(self, training_data, model, batch_size, epochs):
        model.fit((np.array([i[0] for i in training_data]).reshape(-1, 8)), (np.array([i[1] for i in training_data]).reshape(-1, 4)),
                  batch_size=batch_size, epochs=epochs)

        return model


class Agent:
    def __init__(self, initial_episodes, test_episodes, vectors_and_keys, score_requirement):
        self.initial_episodes = initial_episodes
        self.test_episodes = test_episodes
        self.vectors_and_keys = vectors_and_keys
        self.score_requirement = score_requirement

    def get_state(self, game):
        _, _, food, snake, length = game.generate_observations()

        return self.generate_observation(snake, food, length, game)

    def initial_population(self, game):
        training_data = []

        # All scores:
        scores = []

        # Just the scores that met our threshold:
        accepted_scores = []

        # Iterate through however many games we want:
        print('Score Requirement:', self.score_requirement)

        n_games = 0
        total_score = 0

        for _ in range(self.initial_episodes):
            # print('Simulation ', _, " out of ", str(
            #     self.initial_episodes), '\r', end='')
            result = 0

            # Reset game to play again
            game.reset()

            # Moves specifically from this environment:
            game_memory = []

            # Previous observation that we saw
            prev_observation = []
            _, score, food, snake, length = game.generate_observations()

            while game.MAX_STEPS != 0:
                final_move = self.generate_action(
                    snake, length)

                # Do it!
                done, new_score, food, snake, length = game.game_loop(
                    final_move)
                observation = self.generate_observation(
                    snake, food, length, game)

                # Notice that the observation is returned FROM the action
                # so we'll store the previous observation here, pairing
                # the prev observation to the action we'll take.
                if len(prev_observation) > 0:
                    game_memory.append([prev_observation, final_move])

                prev_observation = observation

                if new_score > score:
                    result += 100
                else:
                    result += 1

                if done:
                    break

            if new_score > game.RECORD:
                game.RECORD = new_score

            n_games += 1
            total_score += new_score

            print('Game: ', n_games, 'from: ', self.initial_episodes, 'Score: ',
                  new_score, 'Record: ', game.RECORD)
            # print('Previous observation: ', prev_observation)
            # print('Total score: ', total_score)

            # IF our result is higher than our threshold, we'd like to save
            # every move we made
            # NOTE the reinforcement methodology here
            # All we're doing is reinforcing the result, we're not trying
            # to influence the machine in any way as to HOW that result is
            # reached
            if result >= self.score_requirement:
                accepted_scores.append(result)

                for data in game_memory:
                    # Convert to one-hot (this is the output layer for our neural network)
                    action_sample = [0, 0, 0, 0]
                    action_sample[data[1]] = 1
                    output = action_sample

                    # Saving our training data
                    training_data.append([data[0], output])

            # Save overall scores
            scores.append(result)

        # Save training data to .npy
        # np.save('training_data/training_data' +
        #         str(datetime.now().strftime("%Y%m%d%H%M%S")) + '.npy', training_data)

        return training_data

    def generate_action(self, snake, length):
        action = random.randint(0, 2) - 1
        final_move = self.get_game_action(snake, action, length)

        return final_move

    def get_game_action(self, snake, action, length):
        snake_direction_vector = self.get_snake_direction_vector(snake, length)
        new_direction_vector = snake_direction_vector

        if action == -1:
            new_direction_vector = self.turn_vector_to_the_left(
                snake_direction_vector)
        elif action == 1:
            new_direction_vector = self.turn_vector_to_the_right(
                snake_direction_vector)

        for pair in self.vectors_and_keys:
            if pair[0] == new_direction_vector.tolist():
                game_action = pair[1]

                return game_action

    def generate_observation(self, snake, food, length, game):
        snake_direction_vector = self.get_snake_direction_vector(snake, length)
        food_direction_vector = self.get_food_direction_vector(
            snake, food, length)
        check_left = self.is_direction_blocked(
            snake, self.turn_vector_to_the_left(snake_direction_vector), length, game)
        check_front = self.is_direction_blocked(
            snake, snake_direction_vector, length, game)
        check_right = self.is_direction_blocked(
            snake, self.turn_vector_to_the_right(snake_direction_vector), length, game)
        angle, food_direction_vector_normalized, snake_direction_vector_normalized = self.get_angle(
            snake_direction_vector, food_direction_vector, game)

        return np.array([int(check_left), int(check_front), int(check_right), food_direction_vector_normalized[0], snake_direction_vector_normalized[0], food_direction_vector_normalized[1], snake_direction_vector_normalized[1], angle])

    def get_snake_direction_vector(self, snake, length):
        return np.array(snake[length - 1]) - np.array(snake[length - 2])

    def get_food_direction_vector(self, snake, food, length):
        return np.array(food) - np.array(snake[length - 1])

    def is_direction_blocked(self, snake, direction, length, game):
        point = np.array(snake[length - 1]) + np.array(direction)

        return point.tolist() in snake[:-1] or point[0] < 0 or point[1] < 0 or point[0] >= game.DISPLAY_WIDHT or point[1] >= game.DISPLAY_HEIGHT

    def turn_vector_to_the_left(self, vector):
        return np.array([-vector[1], vector[0]])

    def turn_vector_to_the_right(self, vector):
        return np.array([vector[1], -vector[0]])

    def get_angle(self, snake_direction_vector, food_direction_vector, game):
        norm_of_apple_direction_vector = np.linalg.norm(food_direction_vector)
        norm_of_snake_direction_vector = np.linalg.norm(snake_direction_vector)

        if norm_of_apple_direction_vector == 0:
            norm_of_apple_direction_vector = game.SNAKE_BLOCK

        if norm_of_snake_direction_vector == 0:
            norm_of_snake_direction_vector = game.SNAKE_BLOCK

        food_direction_vector_normalized = food_direction_vector / \
            norm_of_apple_direction_vector
        snake_direction_vector_normalized = snake_direction_vector / \
            norm_of_snake_direction_vector
        angle = m.atan2(food_direction_vector_normalized[1] * snake_direction_vector_normalized[0] - food_direction_vector_normalized[0] * snake_direction_vector_normalized[1],
                        food_direction_vector_normalized[1] * snake_direction_vector_normalized[1] + food_direction_vector_normalized[0] * snake_direction_vector_normalized[0]) / m.pi

        return angle, food_direction_vector_normalized, snake_direction_vector_normalized

    def test_model(self, model, game, helper):
        n_games = 0
        total_score = 0
        start_time = str(datetime.now().strftime("%Y%m%d%H%M%S"))

        for _ in range(self.test_episodes):
            game.reset()
            prev_observation = self.get_state(game)

            while game.MAX_STEPS != 0:
                final_move = np.argmax(np.array(model.predict(
                    np.array(prev_observation).reshape(-1, 8))))

                done, new_score, food, snake, length = game.game_loop(
                    final_move)
                observation = self.generate_observation(
                    snake, food, length, game)

                prev_observation = observation

                if done:
                    break

            if new_score > game.RECORD:
                game.RECORD = new_score

            n_games += 1
            total_score += new_score

            print('Game: ', n_games, 'from: ', self.test_episodes, 'Score: ',
                  new_score, 'Record: ', game.RECORD)
            # print('Previous observation: ', prev_observation)
            # print('Total score: ', total_score)

            if OPTIMIZATION == False and LOAD_WEIGHTS == True:
                helper.write_result_to_list(n_games, new_score)
            elif OPTIMIZATION == False and LOAD_WEIGHTS == False:
                helper.write_result_to_list(n_games, new_score)

        if OPTIMIZATION and LOAD_WEIGHTS == False:
            return total_score
        elif OPTIMIZATION == False and LOAD_WEIGHTS == False:
            helper.write_result_to_csv()
            self.save_test_logs(start_time, game.RECORD, total_score)
        else:
            helper.write_result_to_csv()
            self.save_test_logs(start_time, game.RECORD, total_score)

    def save_test_logs(self, start_time, record_score, total_score):
        with open('logs/test_' + str(datetime.now().strftime("%Y%m%d%H%M%S")) + '.txt', 'a') as f:
            f.write(str('start_time{}_record_score{}_total_score{}'.format(
                start_time, record_score, total_score)) + '\n')
            f.write('Values: {start_time: ' + str(start_time) + ', record_score: ' + str(record_score) +
                    ', total_score: ' + str(total_score) + '}\n')


def main(model):
    # Initialize game
    game = SnakeGame()

    # Initial and test episodes
    initial_episodes = 100000
    test_episodes = 10000
    # Score requirement
    score_requirement = 50

    # Snake move vectors
    vectors_and_keys = [[[-game.SNAKE_BLOCK, 0], 0],  # LEFT
                        [[game.SNAKE_BLOCK, 0], 1],  # RIGHT
                        [[0, -game.SNAKE_BLOCK], 2],  # UP
                        [[0, game.SNAKE_BLOCK], 3]]  # DOWN

    # Initialize helper
    helper = Helper()

    # Initialize agent
    agent = Agent(initial_episodes, test_episodes,
                  vectors_and_keys, score_requirement)

    if OPTIMIZATION and LOAD_WEIGHTS == False:
        if model is None:
            # Generate training data
            training_data = agent.initial_population(game)

            return training_data
        else:
            # Test model after generate training data
            result = agent.test_model(model, game, helper)

            return result
    elif OPTIMIZATION == False and LOAD_WEIGHTS == False:
        # Initialize optimization
        optimization = Optimization()

        # Need to manually write optimal hyperparams
        best_para = {
            'no_of_layers': 4,
            'no_of_neurons': 64,
            'lr': 0.001,
            'batch_size': 192,
            'epochs': 7
        }

        # Hyperparams retyping
        no_of_layers = int(best_para['no_of_layers'])
        no_of_neurons = best_para['no_of_neurons']
        lr = best_para['lr']
        batch_size = int(best_para['batch_size'])
        epochs = int(best_para['epochs'])

        # Initialize nn
        neural_network = NeuralNetwork(no_of_layers, no_of_neurons, lr)
        model = neural_network.model()

        # Generate training data
        training_data = agent.initial_population(game)

        # Train model
        model = optimization.train_model(
            training_data, model, batch_size, epochs)

        # Test loaded model with optimal hyperparams
        agent.test_model(model, game, helper)

        # Save weights
        neural_network.save_weights()
    else:
        # Need to manually write optimal hyperparams
        best_para = {
            'no_of_layers': 4,
            'no_of_neurons': 64,
            'lr': 0.001,
            'batch_size': 192,
            'epochs': 7
        }

        # Hyperparams retyping
        no_of_layers = int(best_para['no_of_layers'])
        no_of_neurons = best_para['no_of_neurons']
        lr = best_para['lr']

        # Initialize nn
        neural_network = NeuralNetwork(no_of_layers, no_of_neurons, lr)
        model = neural_network.model()
        # Load model which have optimal hyperparams
        neural_network.load_weights_()

        # Test loaded model with optimal hyperparams
        agent.test_model(model, game, helper)


if __name__ == "__main__":
    if OPTIMIZATION and LOAD_WEIGHTS == False:
        # Initialize optimization
        optimization = Optimization()

        # Hyperparams
        search_space = {
            'no_of_layers': np.arange(2, 6, 1),
            'no_of_neurons': np.arange(32, 256, 32),
            'lr': np.array([0.01, 0.001, 0.0001, 0.00001]),
            'batch_size': np.arange(32, 256, 32),
            'epochs': np.arange(1, 10, 1)
        }

        # Optimization algorithms
        # opt = HillClimbingOptimizer(search_space)
        opt = StochasticHillClimbingOptimizer(search_space)

        # Run optimization for the N of iterations
        opt.search(optimization.optimize, n_iter=100)
    elif OPTIMIZATION == False and LOAD_WEIGHTS == False:
        # Run training nn with optimized hyperparams
        main(None)
    else:
        # Run game with optimized hyperparams
        main(None)
