import matplotlib.pyplot as plt

data = [(-3, 1), (-2,1), (-1,-1), (0, 1), (1,-1), (2,1), (3,1)]
x_transform_pos = []
x_transform_neg = []
y_transform_pos = []
y_transform_neg = []

def transform(part):
    for pair in data:
        x,y = pair
        if part == 1:
            new_y = -(8/3) * (x ** 2) + (2/3) * (x ** 4)
        else:
            new_y = -4 * (x ** 2) + (1/2) * (x ** 4)

        if y == 1:
            x_transform_pos.append(x)
            y_transform_pos.append(new_y)
        if y == -1:
            x_transform_neg.append(x)
            y_transform_neg.append(new_y)

transform(1)

plt.scatter(x_transform_pos, y_transform_pos)
plt.scatter(x_transform_neg, y_transform_neg)
plt.ylim([-5, 35])
plt.grid()
# center y-axis
ax = plt.gca()
ax.spines['left'].set_position('zero')
ax.spines['right'].set_color('none')
ax.spines['bottom'].set_position('zero')
ax.spines['top'].set_color('none')
# max margin classifier
plt.axhline(y=-1, color='r', linestyle='-')
plt.show()

transform(8)

plt.scatter(x_transform_pos, y_transform_pos)
plt.scatter(x_transform_neg, y_transform_neg)
plt.ylim([-10, 10])
plt.grid()
# center y-axis
ax = plt.gca()
ax.spines['left'].set_position('zero')
ax.spines['right'].set_color('none')
ax.spines['bottom'].set_position('zero')
ax.spines['top'].set_color('none')
plt.show()