import copy
import random
import math

class TeekoPlayer:
    """ An object representation for an AI game player for the game Teeko.
    """
    pieces = ['b', 'r']
    max_depth = 3

    def __init__(self):
        """ Initializes a TeekoPlayer object by randomly selecting red or black as its
        piece color.
        """
        self.board = [[' ' for j in range(5)] for i in range(5)]
        self.my_piece = random.choice(self.pieces)
        self.opp = self.pieces[0] if self.my_piece == self.pieces[1] else self.pieces[1]

    def make_move(self, state):
        """ 
        Selects a (row, col) space for the next move. You may assume that whenever
        this function is called, it is this player's turn to move.

        Args:
            state (list of lists): should be the current state of the game as saved in
                this TeekoPlayer object. Note that this is NOT assumed to be a copy of
                the game state and should NOT be modified within this method (use
                place_piece() instead). Any modifications (e.g. to generate successors)
                should be done on a deep copy of the state.

                In the "drop phase", the state will contain less than 8 elements which
                are not ' ' (a single space character).

        Return:
            move (list): a list of move tuples such that its format is
                    [(row, col), (source_row, source_col)]
                where the (row, col) tuple is the location to place a piece and the
                optional (source_row, source_col) tuple contains the location of the
                piece the AI plans to relocate (for moves after the drop phase). In
                the drop phase, this list should contain ONLY THE FIRST tuple.
        """

        # generate all successors for us and use minimax (max at root)
        successors = self.succ(state, self.my_piece)

        # Fallback: if somehow no legal successors, just drop on first empty cell
        if not successors:
            for r in range(5):
                for c in range(5):
                    if state[r][c] == ' ':
                        return [(r, c)]

        best_val = -math.inf
        best_moves = []

        for move, succ_state in successors:
            # after our move it is opponent's turn (min node)
            val = self.min_value(succ_state, 1)
            if val > best_val + 1e-9:
                best_val = val
                best_moves = [move]
            elif abs(val - best_val) <= 1e-9:
                best_moves.append(move)

        # break ties randomly so AI isn't totally deterministic
        chosen_move = random.choice(best_moves)
        return chosen_move

    def succ(self, state, my_piece): 
        """
        Generate a list of valid successors for the current game state
        for the given piece (my_piece).

        Returns:
            list of (move, new_state) pairs where move has the proper format:
            - drop phase:  [(row, col)]
            - move phase:  [(row, col), (source_row, source_col)]
        """
        successors = []

        # Count pieces to see whether we are in drop phase (<8 pieces total)
        piece_count = sum(1 for row in state for cell in row if cell != ' ')
        drop_phase = piece_count < 8

        if drop_phase:
            # Drop phase: can place on any empty cell
            for r in range(5):
                for c in range(5):
                    if state[r][c] == ' ':
                        new_state = copy.deepcopy(state)
                        new_state[r][c] = my_piece
                        move = [(r, c)]
                        successors.append((move, new_state))
        else:
            # Move phase: move one of my_piece's markers to an adjacent empty cell
            for r in range(5):
                for c in range(5):
                    if state[r][c] == my_piece:
                        for dr in (-1, 0, 1):
                            for dc in (-1, 0, 1):
                                if dr == 0 and dc == 0:
                                    continue
                                nr, nc = r + dr, c + dc
                                # no wrap-around; must stay on board
                                if 0 <= nr < 5 and 0 <= nc < 5 and state[nr][nc] == ' ':
                                    new_state = copy.deepcopy(state)
                                    new_state[r][c] = ' '
                                    new_state[nr][nc] = my_piece
                                    move = [(nr, nc), (r, c)]
                                    successors.append((move, new_state))

        return successors
    
    def opponent_move(self, move):
        """ Validates the opponent's next move against the internal board representation.
        You don't need to touch this code.

        Args:
            move (list): a list of move tuples such that its format is
                    [(row, col), (source_row, source_col)]
                where the (row, col) tuple is the location to place a piece and the
                optional (source_row, source_col) tuple contains the location of the
                piece the AI plans to relocate (for moves after the drop phase). In
                the drop phase, this list should contain ONLY THE FIRST tuple.
        """
        # validate input
        if len(move) > 1:
            source_row = move[1][0]
            source_col = move[1][1]
            if source_row != None and self.board[source_row][source_col] != self.opp:
                self.print_board()
                print(move)
                raise Exception("You don't have a piece there!")
            if abs(source_row - move[0][0]) > 1 or abs(source_col - move[0][1]) > 1:
                self.print_board()
                print(move)
                raise Exception('Illegal move: Can only move to an adjacent space')
        if self.board[move[0][0]][move[0][1]] != ' ':
            raise Exception("Illegal move detected")
        # make move
        self.place_piece(move, self.opp)

    def place_piece(self, move, piece):
        """ Modifies the board representation using the specified move and piece

        Args:
            move (list): a list of move tuples such that its format is
                    [(row, col), (source_row, source_col)]
                where the (row, col) tuple is the location to place a piece and the
                optional (source_row, source_col) tuple contains the location of the
                piece the AI plans to relocate (for moves after the drop phase). In
                the drop phase, this list should contain ONLY THE FIRST tuple.

                This argument is assumed to have been validated before this method
                is called.
            piece (str): the piece ('b' or 'r') to place on the board
        """
        if len(move) > 1:
            self.board[move[1][0]][move[1][1]] = ' '
        self.board[move[0][0]][move[0][1]] = piece

    def print_board(self):
        """ Formatted printing for the board """
        for row in range(len(self.board)):
            line = str(row)+": "
            for cell in self.board[row]:
                line += cell + " "
            print(line)
        print("   A B C D E")

    def _all_winning_lines(self):
        """
        Internal helper: return all winning “lines”:
        - All 4-long horiz/vert/diag segments
        - All 2x2 boxes
        Each returned line is a list of 4 (row,col) coordinates.
        """
        lines = []

        # Horizontal 4-in-a-row
        for r in range(5):
            for c in range(2):  # starting columns 0,1 -> cells c..c+3
                line = [(r, c + i) for i in range(4)]
                lines.append(line)

        # Vertical 4-in-a-row
        for c in range(5):
            for r in range(2):  # starting rows 0,1 -> rows r..r+3
                line = [(r + i, c) for i in range(4)]
                lines.append(line)

        # Diagonal down-right (NW -> SE)
        for r in range(2):
            for c in range(2):
                line = [(r + i, c + i) for i in range(4)]
                lines.append(line)

        # Diagonal down-left (NE -> SW)
        for r in range(2):
            for c in range(3, 5):  # start at columns 3,4
                line = [(r + i, c - i) for i in range(4)]
                lines.append(line)

        # 2x2 boxes (also winning patterns)
        for r in range(4):
            for c in range(4):
                line = [
                    (r, c),
                    (r, c + 1),
                    (r + 1, c),
                    (r + 1, c + 1),
                ]
                lines.append(line)

        return lines
    
    def heuristic_game_value(self, state):
        """ 
        Define the heuristic game value of the current board state taking into account players
        and opponents

        Args:
        state (list of lists): either the current state of the game as saved in
            this TeekoPlayer object, or a generated successor state.

        Returns:
            float heuristic_val (heuristic computed for the game state), in [-1,1]
        """

        # if terminal, just return exact game value
        gv = self.game_value(state)
        if gv != 0:
            return float(gv)

        lines = self._all_winning_lines()
        my_best = 0
        opp_best = 0

        # Look at all potential winning lines; for each one, if it contains only
        # one color (plus blanks), count how many of that color occupy it.
        for line in lines:
            my_count = 0
            opp_count = 0
            for r, c in line:
                if state[r][c] == self.my_piece:
                    my_count += 1
                elif state[r][c] == self.opp:
                    opp_count += 1

            # If both players occupy the line, it's blocked for both
            if my_count > 0 and opp_count > 0:
                continue

            my_best = max(my_best, my_count)
            opp_best = max(opp_best, opp_count)

        # Simple difference heuristic, scaled to [-1,1]
        if my_best == 0 and opp_best == 0:
            heuristic_val = 0.0
        else:
            heuristic_val = (my_best - opp_best) / 4.0

        # Small central-control bonus: encourage using the center area
        center_squares = [(2, 2), (1, 2), (2, 1), (2, 3), (3, 2)]
        center_score = 0.0
        for r, c in center_squares:
            if state[r][c] == self.my_piece:
                center_score += 0.02
            elif state[r][c] == self.opp:
                center_score -= 0.02

        heuristic_val += center_score

        # Clamp final value to [-1, 1]
        heuristic_val = max(-1.0, min(1.0, heuristic_val))
        return heuristic_val
 
    def game_value(self, state):
        """ 
        Checks the current board status for a win condition

        Args:
        state (list of lists): either the current state of the game as saved in
            this TeekoPlayer object, or a generated successor state.

        Returns:
            int: 1 if this TeekoPlayer wins, -1 if the opponent wins, 0 if no winner
        """
        lines = self._all_winning_lines()

        # Check for our win and opponent win separately
        for piece, score in ((self.my_piece, 1), (self.opp, -1)):
            for line in lines:
                if all(state[r][c] == piece for r, c in line):
                    return score
        
        return 0  # no winner yet
    
    def max_value(self, state, depth):
        """
        Helper function for minimax: max node value for self.my_piece.
        """
        gv = self.game_value(state)
        # If terminal or depth limit reached, evaluate heuristically
        if gv != 0 or depth >= self.max_depth:
            return self.heuristic_game_value(state)

        v = -math.inf
        successors = self.succ(state, self.my_piece)
        if not successors:
            return self.heuristic_game_value(state)

        for move, succ_state in successors:
            v = max(v, self.min_value(succ_state, depth + 1))
        return v

    def min_value(self, state, depth):
        """
        Helper function for minimax: min node value for self.opp.
        """
        gv = self.game_value(state)
        if gv != 0 or depth >= self.max_depth:
            return self.heuristic_game_value(state)

        v = math.inf
        successors = self.succ(state, self.opp)
        if not successors:
            return self.heuristic_game_value(state)

        for move, succ_state in successors:
            v = min(v, self.max_value(succ_state, depth + 1))
        return v



############################################################################
#
# THE FOLLOWING CODE IS FOR SAMPLE GAMEPLAY ONLY
#
############################################################################
def main():
    print('Hello, this is Samaritan')
    ai = TeekoPlayer()
    piece_count = 0
    turn = 0

    # drop phase
    while piece_count < 8 and ai.game_value(ai.board) == 0:

        # get the player or AI's move
        if ai.my_piece == ai.pieces[turn]:
            ai.print_board()
            move = ai.make_move(ai.board)
            ai.place_piece(move, ai.my_piece)
            print(ai.my_piece+" moved at "+chr(move[0][1]+ord("A"))+str(move[0][0]))
        else:
            move_made = False
            ai.print_board()
            print(ai.opp+"'s turn")
            while not move_made:
                player_move = input("Move (e.g. B3): ")
                while player_move[0] not in "ABCDE" or player_move[1] not in "01234":
                    player_move = input("Move (e.g. B3): ")
                try:
                    ai.opponent_move([(int(player_move[1]), ord(player_move[0])-ord("A"))])
                    move_made = True
                except Exception as e:
                    print(e)

        # update the game variables
        piece_count += 1
        turn += 1
        turn %= 2

    # move phase - can't have a winner until all 8 pieces are on the board
    while ai.game_value(ai.board) == 0:

        # get the player or AI's move
        if ai.my_piece == ai.pieces[turn]:
            ai.print_board()
            move = ai.make_move(ai.board)
            ai.place_piece(move, ai.my_piece)
            print(ai.my_piece+" moved from "+chr(move[1][1]+ord("A"))+str(move[1][0]))
            print("  to "+chr(move[0][1]+ord("A"))+str(move[0][0]))
        else:
            move_made = False
            ai.print_board()
            print(ai.opp+"'s turn")
            while not move_made:
                move_from = input("Move from (e.g. B3): ")
                while move_from[0] not in "ABCDE" or move_from[1] not in "01234":
                    move_from = input("Move from (e.g. B3): ")
                move_to = input("Move to (e.g. B3): ")
                while move_to[0] not in "ABCDE" or move_to[1] not in "01234":
                    move_to = input("Move to (e.g. B3): ")
                try:
                    ai.opponent_move([(int(move_to[1]), ord(move_to[0])-ord("A")),
                                    (int(move_from[1]), ord(move_from[0])-ord("A"))])
                    move_made = True
                except Exception as e:
                    print(e)

        # update the game variables
        turn += 1
        turn %= 2

    ai.print_board()
    if ai.game_value(ai.board) == 1:
        print("AI wins! Game over.")
    else:
        print("You win! Game over.")


if __name__ == "__main__":
    main()