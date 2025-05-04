import matplotlib.pyplot as plt
import rospy
import rospkg

def read_values_from_file(filename):
    with open(filename, 'r') as file:
        rospy.loginfo(str(line.strip() for line in file))
        values = [float(line.strip()) for line in file]
        
    return values

def plot_values(values):
    x_values = range(len(values))  # Incremental x values
    plt.plot(x_values, values, marker='o', linestyle='-')
    plt.xlabel("Index")
    plt.ylabel("Value")
    plt.title("Plot of Values from File")
    plt.grid()
    plt.show()

def main():
    #rospy.init_node('tiago_DDPG', anonymous=True, log_level=rospy.INFO)
    rospack = rospkg.RosPack()
    pkg_path = rospack.get_path('tiago_navigation') + '/train_results'
    filename = pkg_path + "/reward.txt"  # Change this to your file name
    values = read_values_from_file(filename)
    plot_values(values)

if __name__ == "__main__":
    main()
