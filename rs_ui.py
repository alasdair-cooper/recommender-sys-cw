import curses
from curses.textpad import Textbox, rectangle

class Program:
    def main(self, stdscr):
        self.OPTIONS = [("BROWSE", self.browse), ("RECOMMENDATIONS", self.recommend), ("HELP", self.display_help), ("EXIT", exit)]
        self.HELP = ["BROWSE lets you view various venues from a selection.", "RECOMMENDATIONS provides you with a selection of venues based on your browsing habits."]

        self.stdscr = stdscr
        self.lines = []

        self.y_max, self.x_max = self.stdscr.getmaxyx()
        self.stdscr.leaveok(True)
                
        while True:
            self.command_lines()
            self.redraw()
            self.manage_command(int(self.take_input(f"Enter a number between 1 and {len(self.OPTIONS)}")))

    def redraw(self):
        for i, line in enumerate(self.lines[-self.y_max+3:]):
            self.stdscr.addstr(1+i, 1, line)
            self.stdscr.clrtoeol()

        rectangle(self.stdscr, 0, 0, self.y_max-2, self.x_max-2)
        self.stdscr.refresh()

    def print_lines(self, text: list):
        self.lines = [""] * len(self.lines)
        index = 0
        while index < len(text):
            self.lines[index] = text[index]
            index += 1
        print(self.lines)
        self.redraw()

    def take_input(self, instruction):
        input_string = ""
        while not ':' in input_string: 
            input_string = ""
            input_box = Textbox(curses.newwin(1, self.x_max-2, self.y_max-1, 1))
            for char in "%s: " % instruction:
                input_box.do_command(char)
            input_string = input_box.edit()
        input_string = input_string.split(": ", 1)[1].strip()
        self.redraw()
        return input_string

    def enter_input(self):
        self.take_input("Press enter to continue")

    def browse(self):
        print()

    def recommend(self):
        print()

    def display_help(self):
        self.print_lines(self.HELP)
        self.enter_input()

    y_max = 0
    x_max = 0

    def command_lines(self):
        self.lines.clear()
        commandIndex = 1
        for option in self.OPTIONS:
            self.lines.append(f"Enter {commandIndex} for {option[0]}.")
            commandIndex += 1

    def manage_command(self, command: int):
        self.OPTIONS[command - 1][1]()

if __name__ == "__main__":
    program = Program()
    curses.wrapper(program.main)