from matplotlib import pyplot as plt

def get_values(filename):
    mean = []
    std = []
    values = []
    with open(filename, 'r') as f:
        lines = f.readlines()
        time = float(lines[-1][0])
        lines = lines[:-1]
        for line in lines:
            line = line[1:-1].split(',')
            mean.append(float(line[0]))
            std.append(float(line[1]))
            values.append(float(line[2][2:-2]))
    return mean, std, values, time

def plot_results(title, values, show=False):
    epochs = [i+1 for i in range(len(values))]
    plt.plot(epochs, values)
    plt.xlabel('Epoch')
    plt.ylabel(title)
    plt.title(title + ' values')
    if show:
        plt.show()
    else:
        plt.savefig(f'../Files/{title}.png')


if __name__ == '__main__':
    mean, std, values, time = get_values('results.txt')
    plot_results('Mean', mean, show=True)
    plot_results('Std', std, show=True)
    plot_results('Function', values, show=True)

