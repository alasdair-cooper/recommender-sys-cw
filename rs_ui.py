from logging import exception
import os
import curses
from tokenize import group
import implicit
from matplotlib.pyplot import axis
from parso import parse
import rs_sampler

os.environ['OPENBLAS_NUM_THREADS'] = '1'

import pandas as pd
import numpy as np

from curses.textpad import Textbox, rectangle
from scipy.sparse import csr_matrix , random, vstack, save_npz, load_npz

class Program:
    def main(self, stdscr):
        self.OPTIONS = [("BROWSE", self.browse), ("RECOMMENDATIONS", self.recommend), ("HELP", self.display_help), ("EXIT", exit)]
        self.HELP = ["BROWSE lets you view various venues from a selection.", "RECOMMENDATIONS provides you with a selection of venues based on your browsing habits."]
        self.FILTERING = ["Use content based filtering.", "Use collaborative based filtering."]
        self.WARNING = ["Date is collected on every input, including selection of items.", "You can exit at any time by entering (  x  )."]

        self.stdscr = stdscr
        self.lines = []

        self.y_max, self.x_max = self.stdscr.getmaxyx()

        # Exception if the user's terminal is too small as curses will be buggy
        # or the content will be incomplete
        # if self.y_max < 20 or self.x_max < 140:
        #   raise Exception("Sorry, your terminal is too small. Please resize and try again.")

        self.print_lines(self.WARNING, quitOption=True)
        self.username = self.take_input("Please enter your name to continue")

        self.df = rs_sampler.initialize_sample(200)

        # Initial matrix of data
        self.vdf = self.df.iloc[:, 14:]

        self.udf = pd.DataFrame(columns=self.vdf.columns)
        self.udf.loc[len(self.udf)] = 0 

        # Inverse DF
        self.idf = pd.DataFrame(columns=self.vdf.columns)
        self.idf.loc[len(self.idf)] = 1 / (1 + self.vdf.sum(axis=0))

        # Normalized matrix of data
        self.vdfNormalized = self.vdf.copy(deep=True)
        self.vdfNormalized["sum"] = self.vdfNormalized.sum(axis=1)
        self.vdfNormalized["sum"] = self.vdfNormalized["sum"]**0.5
        # Divide entries by root of sum of features in row
        self.vdfNormalized = self.vdfNormalized.div(self.vdfNormalized["sum"], axis=0)
        # Remove temp sum column
        self.vdfNormalized.pop("sum")

        for col in self.udf.columns:
            self.udf[col].values[:] = 0

        # Create empty column to store a prediction for each item
        self.df["pred"] = 0
        self.df["pred"] = self.df["pred"].astype(float)

        # Matrix of user interests
        if not os.path.exists(f"./user_files/{self.username}.csv"):
            self.vdf["user"] = 0
            self.vdf["user"].to_csv(f"./user_files/{self.username}.csv")
        else:
            userdf = pd.read_csv(f"./user_files/{self.username}.csv", index_col=0)
            self.vdf = self.vdf.join(userdf)

        if not os.path.exists(f"./users.npz"):
            save_npz(f"./users.npz", csr_matrix(random(199, len(self.vdf), density=0.25)))
        self.users = load_npz(f"./users.npz")
    
        # Fill a column with the index of each row so that we can refer 
        # to the correct row in vdf and vdfNormalized later.
        self.df["index"] = range(0, len(self.df))

        while True:
            self.command_lines()
            self.redraw()
            input = ""
            while True:
                try:
                    input = int(self.take_input())
                    break
                except Exception as e:
                    self.lines.append("")
                    self.lines.append("Invalid input.")
                    self.redraw()
            self.manage_command(input)

    def redraw(self):
        self.stdscr.erase()
        for i, line in enumerate(self.lines[-self.y_max+3:]):
            self.stdscr.addstr(1+i, 1, line)
            self.stdscr.clrtoeol()

        rectangle(self.stdscr, 0, 0, self.y_max-2, self.x_max-2)
        self.stdscr.refresh()
        self.stdscr.leaveok(True)

    def print_lines(self, text = [], lineNumbers = False, menuReturnOption = False, quitOption = False):
        if lineNumbers:
            self.lines = []
            index = 1
            for line in text:
                self.lines.append(f"( {index} ) {line}")
                index += 1
        else:
            self.lines = text
        self.lines.append("")
        if menuReturnOption:
            self.lines.append("(  m  ) Return to the previous menu")
        if quitOption:
            self.lines.append("(  x  ) Quit program")
        self.redraw()

    def take_input(self, instruction = "Please enter an option from above to continue"):
        input_string = ""
        while not ':' in input_string: 
            input_string = ""
            input_box = Textbox(curses.newwin(1, self.x_max-2, self.y_max-1, 1))
            for char in f"{instruction}: ":
                input_box.do_command(char)
            input_string = input_box.edit()
        input_string = input_string.split(": ", 1)[1].strip()
        self.redraw()
        if(input_string == "x"):
            exit()
        return input_string

    def enter_input(self):
        self.take_input("Press enter to continue")

    def get_categories_in_row(self, row: pd.Series):
        row = row.filter(like="category.")
        row = row.replace(0, np.nan)
        row = row.dropna(how="all")
        categories = iter(row.axes[0].values)
        text = ""
        i = 0
        for i in range(0, 2):
            category = next(categories)
            category = category.replace("category.", "")
            if len(category) == 0:
                text += ""
            elif category[-1] == "'":
                text += f"{category[:-1]}, "
            else:
                text += f"{category}, "
        return text[:-2]

    def browse(self):
        while True:
            sample = self.df.sample(5)
            self.quick_peak_businesses(sample)

            while True:
                try:
                    input = self.take_input()
                    if input == "m":
                        return
                    else:
                        input = int(input)
                    break
                except Exception as e:
                    self.lines.append("")
                    self.lines.append("Invalid input.")
                    self.redraw()

            business = sample.iloc[int(input) - 1]

            i = 0
            for index, row in sample.iterrows():
                if i == int(input) - 1:
                    # Modify the 'user' column value if the user shows slight interest
                    self.vdf.iloc[row["index"], -1] += 1
                else:
                    # Modify the 'user' column value if the user ignores this in favour of another
                    self.vdf.iloc[row["index"], -1] -= 1
                i += 1
            inputReturned = self.view_business(business)
            score = 0

            if inputReturned.lower() == "y" or inputReturned.lower() == "yes":
                # Modify the 'user' column value if the user likes the 
                self.vdf.iloc[business["index"], -1] += 2
            elif inputReturned.lower() == "n" or inputReturned.lower() == "no":
                # Modify the 'user' column value if the user dislikes the 
                self.vdf.iloc[business["index"], -1] -= 2 
            elif inputReturned == "m":
                return

            self.calculate_pred()
    
    def quick_peak_businesses(self, df: pd.DataFrame):
        linesToPrint = []  
        df.apply(lambda x: linesToPrint.append(f"{x['name']} in {x['city']} has {x['stars']} stars: {self.get_categories_in_row(x)}"), axis=1)
        self.print_lines(linesToPrint, True, True, True)

    def calculate_pred(self):
        # Separate index as pandas preserves index after sample
        for label, value in self.udf.iloc[0, :].iteritems():
            self.udf[label] = sum(self.vdf.user * self.vdfNormalized[label])
        
        i = 0
        for index, row in self.vdfNormalized.iterrows():
            self.df.pred.iloc[i] = sum(self.udf.iloc[0] * self.idf.iloc[0] * self.vdfNormalized.iloc[i])
            i += 1
        
        self.vdf.user.to_csv(f"./user_files/{self.username}.csv")

    def view_business(self, df: pd.DataFrame):
        linesToPrint = []
        categories = "categories: "
        hours = "open: "
        attributes = "extra info: "
        index = 0
        for label, value in df.iteritems():
            labelSplit = iter(label.split("."))
            prefix = next(labelSplit, "")
            labelWithoutPrefix = next(labelSplit, "")
            if prefix == "category":
                if value == 1:
                    categories += labelWithoutPrefix + ", "
            elif prefix == "hours":
                if value != "" or None:
                    hours += f"{labelWithoutPrefix} {value}, "
            elif prefix == "attributes":
                if value != 0:
                    if labelWithoutPrefix == "RestaurantsTakeOut":
                        attributes += "This business has a takeaway option. "
                    elif labelWithoutPrefix == "RestaurantsDelivery":
                        attributes += "This business delivers. "
                    elif labelWithoutPrefix == "WheelchairAccessible":
                        attributes += "This business is wheelchair accessible. "
                    elif labelWithoutPrefix == "DogsAllowed":
                        attributes += "This business allows dogs. "
            else:
                if label != "index" and label != "pred" and label != "stars":
                    linesToPrint.append(f"{str.capitalize(label.replace('_', ' '))}: {value}")
            index += 1

        linesToPrint.append(categories[:-2])
        linesToPrint.append(hours[:-2] if hours != "open: " else "No opening hours available.")
        linesToPrint.append(attributes)

        linesToPrint.append("")
        linesToPrint.append("( y/n ) Interested? Enter y/Y/YES. Not interested? Enter n/N/NO.")

        self.print_lines(linesToPrint, False, True)
        input = self.take_input()
        return input

    def recommend(self):
        self.print_lines(self.FILTERING, True)
        mode = ""
        while True:
            try:
                mode = self.take_input()
                if mode == "m":
                    return
                else:
                    mode = int(mode)
                break
            except Exception as e:
                self.lines.append("")
                self.lines.append("Invalid input.")
                self.redraw()
        mode = int(mode)
        if mode == 1:
            self.calculate_pred()
            df = self.df[self.df["pred"] != 0]
            grouped = df.sort_values("pred", ascending=False).head(5)
            self.quick_peak_businesses(grouped)
            grouped["stars"] = grouped["stars"].astype(float)
            grouped["stars"] = (grouped["stars"] - grouped["stars"].min()) / (grouped["stars"].max() - grouped["stars"].min())
            grouped["pred"] = (grouped["pred"] - grouped["pred"].min()) / (grouped["pred"].max() - grouped["pred"].min())
            grouped["se"] = grouped.apply(lambda x: (x["pred"] - x["stars"])**2, axis=1)
            rmse = ((1 / len(grouped)) * grouped["se"].sum())**0.5
            print(f"content: {rmse}")
        elif mode == 2:
            model = implicit.als.AlternatingLeastSquares(factors=50)
            currentUser = self.vdf.user.values
            currentUserNormalised = (currentUser - np.min(currentUser)) / (np.max(currentUser) - np.min(currentUser))
            user_items = vstack((csr_matrix(currentUserNormalised), self.users))
            item_user = user_items.T.tocsr()
            model.fit(item_user)
            recommendations = model.recommend(0, user_items[0])
            grouped = self.df[self.df["index"].isin(recommendations[0])].head(5)
            grouped["pred"] = recommendations[1][:5]
            self.quick_peak_businesses(grouped)
            grouped["stars"] = grouped["stars"].astype(float)
            grouped["stars"] = (grouped["stars"] - grouped["stars"].min()) / (grouped["stars"].max() - grouped["stars"].min())
            grouped["pred"] = (grouped["pred"] - grouped["pred"].min()) / (grouped["pred"].max() - grouped["pred"].min())
            grouped["se"] = grouped.apply(lambda x: (x["pred"] - x["stars"])**2, axis=1)
            rmse = ((1 / len(grouped)) * grouped["se"].sum())**0.5
            print(f"collaborative: {rmse}")
        self.enter_input()

    def display_help(self):
        self.print_lines(self.HELP)
        self.enter_input()

    y_max = 0
    x_max = 0

    def command_lines(self):
        linesToPrint = []
        for option in self.OPTIONS:
            linesToPrint.append(option[0])
        self.print_lines(linesToPrint, True)

    def manage_command(self, command: int):
        self.OPTIONS[command - 1][1]()

if __name__ == "__main__":
    program = Program()
    curses.wrapper(program.main)