import graphical

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
import numpy as np

import matplotlib.pyplot as plt
from numpy.lib.npyio import load

# Set this to true if you want to load weights from a pre existing file.
LOAD_W8S = True

BETA = 5e-3
GAMES = 100
GAMMA = 0.99
LEARN_RATE = 1e-9


# NUM INPUTS corresponds to the number of characters that make up a board state.
NUM_INPUTS = 89
# This value is reflective of the limits on the valid moves an agent can make, 
# an agent can't choose to move a tile to the right from column 7
# nor can they choose to move a tile downwards from row 9.
# Once these moves are removed from contention (lines 30 - 36) we are left with 142 possible valid moves.
NUM_ACTIONS = 142

BOARD_ROWS = 10
BOARD_COLUMNS = 8

ACTION_IDX = 0
actions = []
actions_taken = []
#the list of available actions in both directions, total amount of actions = 142.
for i in range(BOARD_ROWS):
    for j in range(BOARD_COLUMNS):
        if i > 0 :
            actions.append((j, i, True))
        if j < (BOARD_COLUMNS - 1):
            actions.append((j, i, False))



# The actor and the critic network share the same root network.
# The input layer of the network is of a shape that fits the 80 inputs that combine to form the game board.
# The following 3 shared layers are densly connected and have a relu activation 
input = Input(shape=(NUM_INPUTS, ))
shared_1 = Dense(1024, activation="relu")(input)
shared_2 = Dense(512, activation="relu")(shared_1)
shared_3 = Dense(256, activation="relu")(shared_2)

# The actor policy layer has an input shape of size NUM_ACTIONS (corresponding to the 142 available valid actions the agent can take), 
# it has a softmax activation that will return the actor determined probability of values of the action space
actor_policy_layer = Dense(NUM_ACTIONS, activation="softmax")(shared_3)

# The critic value layer has a single output that is representative of the rewards predicted from the current board.
critic_value_layer = Dense(1, activation=None)(shared_3)

# Here you can see that the model has one input and two outputs. 
model = Model(inputs=input, outputs=[actor_policy_layer, critic_value_layer])

optimizer = Adam(learning_rate=LEARN_RATE)

# tape definition
tape = tf.GradientTape(persistent=True)

# If true than load the weights from previous training sessions.
if LOAD_W8S:
    model.load_weights('weights')


games_played = 0
action_probs_history = []
final_scores = []
scores_history = []
mean_scores = []
actor_losses = []
critic_losses = []
total_losses = []


# The ai_callback is reponsible for returning moves to the game engine. 
# It is implemented as follows:
# The board argument is passed to parse board to tensor. It returns in the form 
# of a tensor and with the newline characters removed. 
# This tensor is passed to the model and the first argument returned is considered. 
# The probabality value of each action is returned from the model.
# An action is chosen and returned to the game.
def ai_callback(board, score, moves_left):
   
    board_tf = parse_board_to_tensor(board)
    action_probs, _ = model(board_tf)
    global ACTION_IDX 
    ACTION_IDX = np.random.choice(NUM_ACTIONS, p=np.squeeze(action_probs))

    return actions[ACTION_IDX]


# The transition callback is called after every move is made. It is at this point that
def transition_callback(board, move, score_delta, next_board, moves_left):
    
    # The score delta is recoreded for the previous move
    scores_history.append(score_delta)

    # The current and next board are parsed and converted to tensor.
    board = parse_board_to_tensor(board)
    next_board = parse_board_to_tensor(next_board)
    
    reward = tf.convert_to_tensor(score_delta, dtype="float32")

    # The gradient tape keeps track of our gradient related calculations.
    with tape:

        # action_probs is equal to actor policy output 
        # board_value is equal to critic value output
        action_probs, board_value = model(board)
        board_value = tf.squeeze(board_value)

        # Calculates the critic value output for the next board state.
        _, next_board_value = model(next_board)
        next_board_value = tf.squeeze(next_board_value)

        # The global value ACTION_IDX is used to append probabality of the same action that has been played by ai_callback
        action_log_prob = tf.math.log(action_probs[0, ACTION_IDX])
        action_probs_history.append(action_log_prob)

        # Entropy is calculated based on previous action probabilities.
        entropy = 0
        for prob in action_probs[0]:
             entropy += (prob * tf.math.log(prob))
        entropy = -entropy

        # The delta or advantage is calculated, the next board value is not applied to the last move of the game.
        # This is a temporal difference advantage function and is typical of the A2C implementation.
        delta = reward + ((GAMMA * next_board_value) * (int(moves_left>0))) - (board_value * (int(moves_left>0))) 

        # Actor and critic losses are calculated. Entropy is applied to the actor loss to hopefully encourage exploration! 
        actor_loss = (-action_log_prob * delta) - (BETA * entropy)
        critic_loss = (delta**2)*(0.5)

        total_loss = critic_loss - actor_loss

        actor_losses.append(actor_loss)
        critic_losses.append(critic_loss)
        total_losses.append(total_loss)

def end_of_game_callback(boards, scores, moves, final_score):

    # At the end of each game the total_losses is fit to the network, the Adam optimizer applies the gradients recorded by the
    with tape:
        grads = tape.gradient(sum(total_losses), model.trainable_variables)
        optimizer.apply_gradients( zip(grads, model.trainable_variables) )
        tape.reset()

    global games_played
    games_played += 1
    final_scores.append(scores[-1])
    if games_played >= GAMES:
        
        # The weights are saved to continue training in a future run.
        if(LOAD_W8S):
            model.save_weights("weights")

        print("Mean after", GAMES, "games:", np.mean(final_scores))

        x_points = list(range(games_played))
        plt.plot(x_points, final_scores)
        plt.show()

        final_scores.clear()
        mean_scores.clear()

        return False

    total_losses.clear()
    action_probs_history.clear()
    scores_history.clear()

    return True

def parse_board_to_tensor(board):
    board = board.replace('\n', '0')
    board = board.replace('#', '1')
    board = board.replace('a', '2')
    board = board.replace('b', '3')
    board = board.replace('c', '4')
    board = board.replace('d', '5')
    board_np = np.array(list(board))
    board_np = board_np.astype("float32")
    board_tf = tf.convert_to_tensor([board_np])
    # board_tf = tf.expand_dims(board_tf, 0)
    return board_tf

if __name__ == '__main__':
    speedup = 999.0
    g = graphical.Game(ai_callback, transition_callback, end_of_game_callback, speedup)
    g.run()