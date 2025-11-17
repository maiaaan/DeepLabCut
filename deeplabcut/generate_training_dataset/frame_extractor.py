from pathlib import Path
import cv2
import tkinter as tk
from PIL import Image, ImageTk
import os
import sys
import numpy as np
from threading import Thread, Lock
import threading
from queue import Queue
import time

class FrameExtractor:
    def slider_click_start(self, event):
        """Handle slider click start"""
        with self.slider_lock:
            self.is_sliding = True

    def slider_click_end(self, event):
        """Handle slider click end"""
        with self.slider_lock:
            self.is_sliding = False
        # Update frame after sliding is done
        self.slider_changed(self.frame_slider.get())

    def slider_changed(self, value):
        """Handle slider position change"""
        with self.slider_lock:
            if self.is_sliding:
                # Only update the frame counter during sliding
                frame_idx = int(float(value))
                self.counter_label.config(text=f"{frame_idx + 1}/{self.total_frames}")
                return
                
        try:
            frame_idx = int(float(value))
            if frame_idx != self.current_frame_idx:
                self.current_frame_idx = frame_idx
                if self._load_current_frame():
                    self.update_display()
        except ValueError as e:
            print(f"Invalid slider value: {e}")

    def __init__(self, video_path):
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
            
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        
        # Validate video file
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
            
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if self.total_frames <= 0:
            raise ValueError(f"Invalid frame count: {self.total_frames}")
            
        self.current_frame_idx = 0
        
        # Add locks for thread safety
        self.frame_lock = threading.Lock()
        self.slider_lock = threading.Lock()
        self.is_sliding = False
        
        # Pre-calculate root directory and extraction path
        self.root_dir = os.path.dirname(os.path.dirname(os.path.abspath(video_path)))
        file_name = Path(video_path).name.split('.')[0]
        self.extract_dir = os.path.join(self.root_dir, 'labeled-data', file_name)
        os.makedirs(self.extract_dir, exist_ok=True)
        
        # Initialize frame buffer and queue
        self.frame_buffer = {}
        self.buffer_size = 30  # Keep 30 frames in memory
        self.preload_queue = Queue()
        
        # Initialize GUI window and get references
        self.root = tk.Tk()
        self.root.title("Frame Extractor")
        
        # Initialize all GUI components
        self.setup_gui()
        
        # Load initial frame
        self._load_current_frame()
        self.update_display()
        
        # Start preloading thread
        self.preload_thread = Thread(target=self._preload_frames, daemon=True)
        self.preload_thread.start()

    def setup_gui(self):
        """Set up the GUI elements"""
        # Create canvas for image display
        self.canvas = tk.Canvas(self.root, width=800, height=600)
        self.canvas.pack(pady=5)
        
        # Create frame for controls
        controls_frame = tk.Frame(self.root)
        controls_frame.pack(fill=tk.X, padx=5)
        
        # Create slider with bindings
        self.frame_slider = tk.Scale(
            controls_frame, 
            from_=0, 
            to=self.total_frames-1, 
            orient=tk.HORIZONTAL,
            command=self.slider_changed
        )
        self.frame_slider.bind('<Button-1>', self.slider_click_start)
        self.frame_slider.bind('<ButtonRelease-1>', self.slider_click_end)
        self.frame_slider.pack(fill=tk.X, pady=5)
        
        # Create frame navigation controls
        nav_frame = tk.Frame(controls_frame)
        nav_frame.pack(fill=tk.X, pady=5)
        
        # Create frame input
        tk.Label(nav_frame, text="Frame:").pack(side=tk.LEFT)
        self.frame_entry = tk.Entry(nav_frame, width=10)
        self.frame_entry.pack(side=tk.LEFT, padx=5)
        self.frame_entry.bind('<Return>', self.jump_to_frame)
        
        # Create counter label
        self.counter_label = tk.Label(nav_frame, text="0/0")
        self.counter_label.pack(side=tk.LEFT, padx=10)
        
        # Create extract button
        extract_btn = tk.Button(nav_frame, text="Extract Frame (E)", command=self.extract_current_frame)
        extract_btn.pack(side=tk.RIGHT)
        
        # Create range extraction controls
        range_frame = tk.Frame(controls_frame)
        range_frame.pack(fill=tk.X, pady=5)
        
        tk.Label(range_frame, text="Extract range - From:").pack(side=tk.LEFT)
        self.range_start = tk.Entry(range_frame, width=10)
        self.range_start.pack(side=tk.LEFT, padx=5)
        
        tk.Label(range_frame, text="To:").pack(side=tk.LEFT)
        self.range_end = tk.Entry(range_frame, width=10)
        self.range_end.pack(side=tk.LEFT, padx=5)
        
        extract_range_btn = tk.Button(range_frame, text="Extract Range (R)", command=self.extract_range)
        extract_range_btn.pack(side=tk.RIGHT)
        
        # Progress label for range extraction
        self.progress_label = tk.Label(range_frame, text="")
        self.progress_label.pack(side=tk.RIGHT, padx=10)
        
        # Bind keys
        self.root.bind('<Left>', lambda e: self.prev_frame())
        self.root.bind('<Right>', lambda e: self.next_frame())
        self.root.bind('e', lambda e: self.extract_current_frame())
        self.root.bind('r', lambda e: self.extract_range())

    def _preload_frames(self):
        """Background thread for preloading frames"""
        while True:
            try:
                if self.preload_queue.empty():
                    time.sleep(0.01)
                    continue
                    
                frame_idx = self.preload_queue.get()
                if frame_idx not in self.frame_buffer and 0 <= frame_idx < self.total_frames:
                    if self.cap.get(cv2.CAP_PROP_POS_FRAMES) != frame_idx:
                        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                    ret, frame = self.cap.read()
                    if ret:
                        self.frame_buffer[frame_idx] = frame.copy()
                        
                        # Remove old frames from buffer
                        while len(self.frame_buffer) > self.buffer_size:
                            oldest_idx = min(k for k in self.frame_buffer.keys() 
                                          if k != self.current_frame_idx)
                            del self.frame_buffer[oldest_idx]
            except Exception as e:
                print(f"Error in preload thread: {e}")
                time.sleep(0.1)

    def _load_current_frame(self):
        """Load the current frame into buffer if needed"""
        with self.frame_lock:
            if self.current_frame_idx not in self.frame_buffer:
                try:
                    current_pos = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
                    if current_pos != self.current_frame_idx:
                        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_idx)
                    ret, frame = self.cap.read()
                    if ret:
                        self.frame_buffer[self.current_frame_idx] = frame.copy()
                    else:
                        print(f"Failed to read frame {self.current_frame_idx}")
                        return False
                except Exception as e:
                    print(f"Error loading frame: {e}")
                    return False
            return True

    def update_display(self):
        """Update the displayed frame"""
        try:
            # Request frame preloading
            for offset in range(-5, 6):  # Preload 5 frames before and after
                target_idx = self.current_frame_idx + offset
                if target_idx not in self.frame_buffer and 0 <= target_idx < self.total_frames:
                    self.preload_queue.put(target_idx)
            
            frame = self.frame_buffer.get(self.current_frame_idx)
            if frame is not None:
                # Convert frame to RGB for PIL
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Resize frame to fit canvas while maintaining aspect ratio
                height, width = frame_rgb.shape[:2]
                scale = min(800/width, 600/height)
                new_width, new_height = int(width * scale), int(height * scale)
                frame_resized = cv2.resize(frame_rgb, (new_width, new_height))
                
                # Convert to PhotoImage
                self.photo = ImageTk.PhotoImage(image=Image.fromarray(frame_resized))
                self.canvas.delete("all")
                self.canvas.create_image(400, 300, anchor=tk.CENTER, image=self.photo)
                
                # Update counter
                self.counter_label.config(text=f"{self.current_frame_idx + 1}/{self.total_frames}")
        except Exception as e:
            print(f"Error updating display: {e}")

    def next_frame(self):
        """Move to next frame"""
        if self.current_frame_idx < self.total_frames - 1:
            self.current_frame_idx += 1
            if self._load_current_frame():
                self.update_display()
                self.frame_slider.set(self.current_frame_idx)

    def prev_frame(self):
        """Move to previous frame"""
        if self.current_frame_idx > 0:
            self.current_frame_idx -= 1
            if self._load_current_frame():
                self.update_display()
                self.frame_slider.set(self.current_frame_idx)

    def jump_to_frame(self, event=None):
        """Jump to frame number entered in text box"""
        try:
            frame_idx = int(self.frame_entry.get())
            if 0 <= frame_idx < self.total_frames:
                self.current_frame_idx = frame_idx
                self.frame_slider.set(frame_idx)
                if self._load_current_frame():
                    self.update_display()
            else:
                print(f"Frame index out of range: {frame_idx}")
        except ValueError as e:
            print(f"Invalid frame number: {e}")
        
        # Clear focus from entry
        self.root.focus_set()

    def extract_current_frame(self):
        """Extract current frame as PNG"""
        frame = self.frame_buffer.get(self.current_frame_idx)
        if frame is not None:
            try:
                output_path = os.path.join(self.extract_dir, f'img{self.current_frame_idx:05d}.png')
                cv2.imwrite(output_path, frame)
                print(f"Extracted frame to: {output_path}")
            except Exception as e:
                print(f"Error extracting frame: {e}")

    def extract_range(self, event=None):
        """Extract a range of frames as PNGs"""
        try:
            start_frame = int(self.range_start.get())
            end_frame = int(self.range_end.get())
            
            if not (0 <= start_frame < self.total_frames and 0 <= end_frame < self.total_frames):
                print("Frame range out of bounds")
                return
                
            if start_frame > end_frame:
                start_frame, end_frame = end_frame, start_frame
                
            # Start extraction in a separate thread to keep GUI responsive
            extraction_thread = Thread(target=self._extract_range_thread, 
                                    args=(start_frame, end_frame), 
                                    daemon=True)
            extraction_thread.start()
            
        except ValueError:
            print("Please enter valid frame numbers")

    def _extract_range_thread(self, start_frame, end_frame):
        """Thread function for range extraction"""
        try:
            total_frames = end_frame - start_frame + 1
            frames_extracted = 0
            
            for frame_idx in range(start_frame, end_frame + 1):
                # Update progress
                frames_extracted += 1
                progress = (frames_extracted / total_frames) * 100
                self.root.after(0, self.progress_label.config, 
                              {"text": f"Progress: {progress:.1f}%"})
                
                # Seek to frame
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = self.cap.read()
                
                if ret:
                    output_path = os.path.join(self.extract_dir, 
                                             f'img{frame_idx:05d}.png')
                    cv2.imwrite(output_path, frame)
                else:
                    print(f"Failed to read frame {frame_idx}")
                    
            # Clear progress label when done
            self.root.after(2000, self.progress_label.config, {"text": ""})
            print(f"Extracted {frames_extracted} frames")
            
        except Exception as e:
            print(f"Error during range extraction: {e}")
            self.root.after(0, self.progress_label.config, {"text": "Error!"})

    def run(self):
        """Start the GUI main loop"""
        try:
            self.root.mainloop()
        finally:
            self.cap.release()

def frame_extraction_gui(video_path):
    try:
        extractor = FrameExtractor(video_path)
        extractor.run()
    except Exception as e:
        print(f"Error initializing frame extractor: {e}")
        sys.exit(1)

if __name__ == "__main__":
    import argparse
    
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Video frame extraction GUI')
    parser.add_argument('video_path', type=str, help='Path to the video file')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Run GUI
    frame_extraction_gui(args.video_path)