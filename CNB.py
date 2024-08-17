import cv2
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog, messagebox
import os

def binarize_capillary_network(image_path, scale_percent=400, blur_kernel_size=(5, 5)):
    # Load the image in grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Resize the image to increase resolution
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized_image = cv2.resize(image, dim, interpolation=cv2.INTER_CUBIC)

    # Apply Gaussian blur to smooth the image
    blurred_image = cv2.GaussianBlur(resized_image, blur_kernel_size, 0)

    # Convert to binary (0 for white, 1 for black)
    _, binary_image = cv2.threshold(blurred_image, 127, 1, cv2.THRESH_BINARY_INV)

    # Plot the binary matrix
    plt.figure(figsize=(6, 6))
    plt.imshow(binary_image, cmap='gray', interpolation='nearest')
    plt.title('Binary Image (0 for white, 1 for black)')
    plt.axis('off')
    # plt.show()

    return binary_image

def save_binary_matrix_as_csv(binary_matrix, output_path):
    np.savetxt(output_path, binary_matrix, delimiter=',', fmt='%d')
    messagebox.showinfo("Success", f"Binary matrix saved as {output_path}")

def save_binary_image(binary_image, image_output_path):
    cv2.imwrite(image_output_path, binary_image * 255)
    messagebox.showinfo("Success", f"Binary image saved as {image_output_path}")

def open_file():
    file_path = filedialog.askopenfilename(
        title="Select an Image",
        filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff")]
    )
    if file_path:
        binary_matrix = binarize_capillary_network(file_path)
        save_outputs(file_path, binary_matrix)

def save_outputs(image_path, binary_matrix):
    save_dir = os.path.dirname(image_path)
    
    # Save CSV
    csv_file_name = os.path.splitext(os.path.basename(image_path))[0] + "_binary_matrix.csv"
    csv_output_path = os.path.join(save_dir, csv_file_name)
    save_binary_matrix_as_csv(binary_matrix, csv_output_path)
    
    # Save binary image
    image_file_name = os.path.splitext(os.path.basename(image_path))[0] + "_binary_image.jpg"
    image_output_path = os.path.join(save_dir, image_file_name)
    save_binary_image(binary_matrix, image_output_path)

# Create the GUI
root = tk.Tk()
root.title("Capillary Network Binarizer")

frame = tk.Frame(root, padx=20, pady=20)
frame.pack()

label = tk.Label(frame, text="Capillary Network Binarizer", font=("Arial", 16))
label.pack(pady=10)

button = tk.Button(frame, text="Select Image", command=open_file, font=("Arial", 14))
button.pack(pady=10)

root.mainloop()
