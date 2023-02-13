
from tkinter import *
import numpy as np

canvas_width, canvas_height = 800, 900
margin_x, margin_y = 120, 120
num_rows, num_columns = 150, 150
cell_length = round((canvas_width-2*margin_x)/num_rows)

colors = {
    "rich_black": "#111213",
    "eerie_black": "#212224",
    "quick_silver": "#a1a1a1",
    "white": "#ffffff",
    "blue_crayola": "#2f71e8",
    "cyan_process": "#29b9f0",
    }

root = Tk()
root.geometry(f"{canvas_width}x{canvas_height}")
canvas = Canvas(root, bg=colors["rich_black"])
canvas.pack(expand=True, fill=BOTH)
root.bind("<Escape>", lambda e: root.destroy())


stroke_slider = Scale(canvas, from_=1, to=20, orient=HORIZONTAL)
stroke_slider.place(x=350, y=750)

def create_backdrop():
    width = 10
    for i in range(2):
        canvas.create_rectangle(margin_x-width+width*i, margin_y-width*i, (margin_x+num_columns*cell_length)+width-width*i+1, margin_y+num_rows*cell_length+width*i+1, 
                        outline="", fill=colors["blue_crayola"])
    canvas.create_oval(margin_x-width, margin_y-width, margin_x+width, margin_y+width, outline="", fill=colors["blue_crayola"])
    canvas.create_oval(margin_x-width, margin_y-width+num_columns*cell_length, margin_x+width, margin_y+width+num_columns*cell_length, outline="", fill=colors["blue_crayola"])
    canvas.create_oval(margin_x-width+num_rows*cell_length, margin_y-width, margin_x+width+num_rows*cell_length, margin_y+width, outline="", fill=colors["blue_crayola"])
    canvas.create_oval(margin_x-width+num_rows*cell_length, margin_y-width+num_columns*cell_length, margin_x+width+num_rows*cell_length, margin_y+width+num_columns*cell_length, outline="", fill=colors["blue_crayola"])
create_backdrop()


class _cell:
    def __init__(self, x, y):
        self.x1 = x*cell_length+margin_x
        self.y1 = y*cell_length+margin_y
        self.x2 = (x+1)*cell_length+margin_x
        self.y2 = (y+1)*cell_length+margin_y
        self.color = colors["quick_silver"]
        
        self.cell = canvas.create_rectangle(self.x1, self.y1, self.x2, self.y2, fill=self.color, outline=self.color)

    def activate(self):
        self.color = colors["eerie_black"]

    def activate_test(self):
        self.color = colors["cyan_process"]
        
    def update(self):
        canvas.itemconfig(self.cell, fill=self.color, outline=self.color)



cells_map = np.array([[_cell(x, y) for y in range(num_columns)] for x in range(num_rows)])

def activate_by_stroke(cell_anchor_x, cell_anchor_y, test):  
    stroke = stroke_slider.get()

    for i in range(2*stroke+1):    
        for j in range(2*stroke+1):
            x = i-stroke
            y = j-stroke
            try:
                if x**2 + y**2 > stroke:
                    pass
                else:
                    if test == 1:
                        cells_map[cell_anchor_x+round(x)][cell_anchor_y+round(y)].activate_test()
                        cells_map[cell_anchor_x+round(x)][cell_anchor_y+round(y)].update()
                    else:
                        cells_map[cell_anchor_x+round(x)][cell_anchor_y+round(y)].activate()
                        cells_map[cell_anchor_x+round(x)][cell_anchor_y+round(y)].update()
            except:
                pass
        
        
lst_cell_x = np.array([])
lst_cell_y = np.array([])
def activate_cells(event):
    x, y = event.x, event.y

    if margin_x < x < margin_x + num_columns*cell_length and \
       margin_y < y < margin_y + num_rows*cell_length:
        lst_cell_x = np.append(lst_cell_x, x)
        lst_cell_y = np.append(lst_cell_y, y)

        cell_x = int(((x-margin_x) - ((x-margin_x)%cell_length))/cell_length)
        cell_y = int(((y-margin_y) - ((y-margin_y)%cell_length))/cell_length)

        activate_by_stroke(cell_x, cell_y, 0)


def fill_gaps(event):
    lst_x_coords_for_fill = np.array([])
    lst_y_coords_for_fill = np.array([])   
    for i, e in enumerate(lst_cell_x):
        try:
            if abs(lst_cell_x[i+1]-lst_cell_x[i]) > abs(lst_cell_y[i+1]-lst_cell_y[i]) > stroke_slider.get():
                num_steps = round((abs(lst_cell_x[i+1]-lst_cell_x[i])%cell_length)/cell_length)
                for n in range((abs(lst_cell_x[i+1]-lst_cell_x[i]) - num_steps)):
                    print(f"n: {n}")
                    lst_x_coords_for_fill = np.append(lst_x_coords_for_fill, lst_cell_x[i]+n*round((lst_cell_x[i+1]-lst_cell_x[i])/num_steps))
                    lst_y_coords_for_fill = np.append(lst_y_coords_for_fill, lst_cell_y[i]+n*round((lst_cell_x[i+1]-lst_cell_x[i])/num_steps))

            elif abs(lst_cell_y[i+1]-lst_cell_y[i]) >= abs(lst_cell_x[i+1]-lst_cell_x[i]) > stroke_slider.get():
                num_steps = round((abs(lst_cell_y[i+1]-lst_cell_y[i])%cell_length)/cell_length)
                for n in range((abs(lst_cell_y[i+1]-lst_cell_y[i]) - num_steps)):
                    lst_y_coords_for_fill = np.append(lst_y_coords_for_fill, lst_cell_y[i]+n*round((lst_cell_x[i+1]-lst_cell_x[i])/num_steps))
                    lst_x_coords_for_fill = np.append(lst_x_coords_for_fill, lst_cell_y[i]+n*round((lst_cell_y[i+1]-lst_cell_y[i])/num_steps))
        except:
            pass

    for i, e in enumerate(lst_x_coords_for_fill):
        activate_by_stroke(lst_x_coords_for_fill[i], lst_y_coords_for_fill[i], 1)

    lst_cell_x = np.array([])
    lst_cell_y = np.array([])


root.bind("<Button-1>", activate_cells)
root.bind("<B1-Motion>", activate_cells)
root.bind("<ButtonRelease-1>", fill_gaps)


def create_buttons():
    def clear():
        for i in range(num_rows):
            for j in range(num_columns):
                cells_map[i][j].color = colors["quick_silver"]
                cells_map[i][j].update()
                
    clear_button = Button(canvas, text="CLEAR", command=clear)
    clear_button.place(x=650, y=750)
create_buttons()



root.mainloop()