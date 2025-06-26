import tkinter as tk
from tkinter import ttk, messagebox
import cv2
import os
import numpy as np
import glob
from PIL import Image, ImageTk
import threading
from loguru import logger
import argparse
from datetime import datetime

from visualize import *

class ImageViewerGUI:
    def __init__(self, root, dataset_path):
        self.root = root
        self.root.title("Image Viewer")
        self.root.geometry("1000x700")
        
        # Variables
        self.dataset_path  = dataset_path
        self.image_paths   = []
        self.current_index = 0
        self.is_playing    = False
        self.current_image = None
        self.photo         = None
        self.label         = None
        
        # Data segregation
        self.correct_label      = []
        self.incorrect_label    = []
        self.unlabeled_goodIMGS = []
        
        # Setup logging
        self.setup_logging()
        
        # Create GUI elements
        self.create_widgets()
        
    def setup_logging(self):
        logPath = "logs"
        if not os.path.exists(logPath):
            os.makedirs(logPath, exist_ok=True)
            
        logger.add(os.path.join(logPath, "image_viewer.log"), rotation="1 day", level="INFO",
                  retention="60 days", compression="zip",
                  enqueue=True, backtrace=True, diagnose=True, colorize=False)
    
    def create_widgets(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(0, weight=1)
        
        # Image display - made bigger and shifted left
        self.image_label = ttk.Label(main_frame, text="No image loaded\nReady", anchor=tk.W, font=("Arial", 12), foreground="blue")
        self.image_label.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=10, padx=(20, 0))
        
        # Right side buttons - arranged vertically
        right_button_frame = ttk.Frame(main_frame)
        right_button_frame.grid(row=0, column=1, sticky=(tk.N, tk.S), padx=(20, 20), pady=10)
        right_button_frame.grid_rowconfigure(1, weight=1)  # Center the button container
        
        # Container for buttons to center them
        button_container = ttk.Frame(right_button_frame)
        button_container.grid(row=1, column=0)
        
        # Start and Next buttons moved to top of right side
        start_next_frame = ttk.Frame(button_container)
        start_next_frame.pack(pady=5)
        
        self.start_btn = ttk.Button(start_next_frame, text="Start", command=self.start_viewing)
        self.start_btn.pack(side=tk.LEFT, padx=5)
        
        self.next_btn = ttk.Button(start_next_frame, text="Next", command=self.next_image, state=tk.DISABLED)
        self.next_btn.pack(side=tk.LEFT, padx=5)
        
        self.correct_btn = tk.Button(button_container, text="Correct Label", fg="red", state=tk.DISABLED)
        self.correct_btn.pack(pady=5)
        
        self.incorrect_btn = tk.Button(button_container, text="No/Incorrect Label", fg="red", state=tk.DISABLED)
        self.incorrect_btn.pack(pady=5)
        
        self.good_image_btn = tk.Button(button_container, text="Good Image", fg="red", state=tk.DISABLED)
        self.good_image_btn.pack(pady=5)
        
        # Connect buttons to their respective methods
        self.correct_btn.config(command=self.add_to_correct_label)
        self.incorrect_btn.config(command=self.add_to_incorrect_label)
        self.good_image_btn.config(command=self.add_to_good_images)
        
        # Exit button at the bottom - centered
        exit_btn = ttk.Button(main_frame, text="Exit", command=self.exit_gui)
        exit_btn.grid(row=1, column=0, pady=10)
    
    def load_images(self):
        if not os.path.exists(self.dataset_path):
            messagebox.showerror("Error", f"Path does not exist: {self.dataset_path}")
            return False
            
        # Read the dataset_path as a text file containing image paths (one per line)
        with open(self.dataset_path, "r") as f:
            self.image_paths = [line.strip() for line in f if line.strip()]
        
        #self.image_paths = glob.glob(os.path.join(self.dataset_path, "*.jpg"))
        if not self.image_paths:
            messagebox.showerror("Error", "No .jpg images found in the selected directory")
            return False
            
        logger.info(f"Loaded {len(self.image_paths)} images from {self.dataset_path}")
        return True
    
    def start_viewing(self):
        if not self.load_images():
            return
            
        self.current_index = 0
        self.is_playing = True
        self.start_btn.config(state=tk.DISABLED)
        
        # Display the first image
        self.display_current_image()
    
    def display_current_image(self):
        if not self.image_paths or self.current_index >= len(self.image_paths):
            return
            
        self.img_path = self.image_paths[self.current_index]
        label_path = self.img_path.replace("images", "labels").replace(".jpg", ".txt")

        label = np.loadtxt(label_path, dtype=np.float32)
        
        if label.ndim == 1 and len(label)>0:
            label = label[np.newaxis, :]
        
        # Load and resize image
        image = cv2.imread(self.img_path)
        if image is None:
            logger.error(f"Failed to load image: {self.img_path}")
            return
        
        if len(label) > 0:
            image = annotate(image, label)
            
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize image to fit in the GUI (max 600x400)
        height, width = image.shape[:2]
        max_width, max_height = 700, 500
        
        if width > max_width or height > max_height:
            scale = min(max_width / width, max_height / height)
            new_width = int(width * scale)
            new_height = int(height * scale)
            image = cv2.resize(image, (new_width, new_height))
        
        # Convert to PIL Image and then to PhotoImage
        pil_image = Image.fromarray(image)
        self.photo = ImageTk.PhotoImage(pil_image)
        
        # Update GUI in main thread
        self.root.after(0, self.update_image_display, self.img_path)
    
    def update_image_display(self, img_path):
        if self.photo:
            self.image_label.config(image=self.photo)
        else:
            self.image_label.config(image="", text="Failed to load image")
        
        # Enable classification buttons after first image is displayed, keep next disabled
        if self.is_playing:
            self.next_btn.config(state=tk.DISABLED)
            self.correct_btn.config(state=tk.NORMAL)
            self.incorrect_btn.config(state=tk.NORMAL)
            self.good_image_btn.config(state=tk.NORMAL)
    
    def next_image(self):
        if self.image_paths and self.is_playing:
            self.current_index = (self.current_index + 1) % len(self.image_paths)
            self.display_current_image()
            
            # Check if we've reached the end
            if self.current_index == len(self.image_paths) - 1:
                # Disable all buttons except exit when all data is finished
                self.next_btn.config(state=tk.DISABLED)
                self.correct_btn.config(state=tk.DISABLED)
                self.incorrect_btn.config(state=tk.DISABLED)
                self.good_image_btn.config(state=tk.DISABLED)
    
    def enable_next_button(self):
        """Enable next button and disable classification buttons"""
        self.next_btn.config(state=tk.NORMAL)
        self.correct_btn.config(state=tk.DISABLED)
        self.incorrect_btn.config(state=tk.DISABLED)
        self.good_image_btn.config(state=tk.DISABLED)
    
    def add_to_correct_label(self):
        """Add current image path to correct_label list"""
        if hasattr(self, 'img_path') and self.img_path:
            img_path_with_newline = self.img_path + "\n"
            if img_path_with_newline not in self.correct_label:
                self.correct_label.append(img_path_with_newline)
                logger.info(f"Added to correct_label: {self.img_path}")
                logger.opt(colors=True).info(f"<green>correct_label list: {self.correct_label}</green>")
        
        # Enable next button and disable classification buttons
        self.enable_next_button()
    
    def add_to_incorrect_label(self):
        """Add current image path to incorrect_label list"""
        if hasattr(self, 'img_path') and self.img_path:
            img_path_with_newline = self.img_path + "\n"
            if img_path_with_newline not in self.incorrect_label:
                self.incorrect_label.append(img_path_with_newline)
                logger.info(f"Added to incorrect_label: {self.img_path}")
                logger.opt(colors=True).info(f"<green>incorrect_label list: {self.incorrect_label}</green>")
        
        # Enable next button and disable classification buttons
        self.enable_next_button()
    
    def add_to_good_images(self):
        """Add current image path to unlabeled_goodIMGS list"""
        if hasattr(self, 'img_path') and self.img_path:
            img_path_with_newline = self.img_path + "\n"
            if img_path_with_newline not in self.unlabeled_goodIMGS:
                self.unlabeled_goodIMGS.append(img_path_with_newline)
                logger.info(f"Added to unlabeled_goodIMGS: {self.img_path}")
                logger.opt(colors=True).info(f"<green>unlabeled_goodIMGS list: {self.unlabeled_goodIMGS}</green>")
        
        # Enable next button and disable classification buttons
        self.enable_next_button()
    
    def save_lists_to_files(self):
        """Save the respective lists as text files in the current directory"""
        try:
            # Generate timestamp for file names
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            saved_files = []
            skipped_files = []
            
            # Save correct_label list only if not empty
            if self.correct_label:
                filename = f"correct_label_{timestamp}.txt"
                with open(filename, "w") as f:
                    f.writelines(self.correct_label)
                logger.info(f"Saved {len(self.correct_label)} paths to {filename}")
                saved_files.append(filename)
            else:
                logger.info("correct_label list is empty - skipping correct_label file")
                skipped_files.append("correct_label")
            
            # Save incorrect_label list only if not empty
            if self.incorrect_label:
                filename = f"incorrect_label_{timestamp}.txt"
                with open(filename, "w") as f:
                    f.writelines(self.incorrect_label)
                logger.info(f"Saved {len(self.incorrect_label)} paths to {filename}")
                saved_files.append(filename)
            else:
                logger.info("incorrect_label list is empty - skipping incorrect_label file")
                skipped_files.append("incorrect_label")
            
            # Save unlabeled_goodIMGS list only if not empty
            if self.unlabeled_goodIMGS:
                filename = f"unlabeled_goodIMGS_{timestamp}.txt"
                with open(filename, "w") as f:
                    f.writelines(self.unlabeled_goodIMGS)
                logger.info(f"Saved {len(self.unlabeled_goodIMGS)} paths to {filename}")
                saved_files.append(filename)
            else:
                logger.info("unlabeled_goodIMGS list is empty - skipping unlabeled_goodIMGS file")
                skipped_files.append("unlabeled_goodIMGS")
            
            # Summary logging
            if saved_files:
                logger.info(f"Successfully saved files: {', '.join(saved_files)}")
            if skipped_files:
                logger.info(f"Skipped empty files: {', '.join(skipped_files)}")
            
            logger.info("File saving operation completed")
            
        except Exception as e:
            logger.error(f"Error saving lists to files: {e}")
    
    def exit_gui(self):
        """Gracefully exit the GUI"""
        logger.info("Exiting Image Viewer GUI")
        
        # Save lists before exiting (only non-empty ones)
        self.save_lists_to_files()
        
        self.root.quit()
        self.root.destroy()

def main():
    parser = argparse.ArgumentParser(description="Image Viewer GUI for dataset visualization")
    parser.add_argument("--path", type=str, 
                       default="/data/Datasets/Golf_project/dataset_split/train1.txt",
                       help="Path to the dataset directory containing images")
    args = parser.parse_args()
    
    root = tk.Tk()
    app = ImageViewerGUI(root, args.path)
    root.mainloop()

if __name__ == "__main__":
    main() 