import './App.css';
import React, { useState } from "react";
import axios from "axios";

function App() {
  const [review, setReview] = useState("");
  const [result, setResult] = useState("");

  const checkReview = async () => {
    if (!review.trim()) {
      setResult("Please enter a review.");
      return;
    }

    try {
      const response = await axios.post("http://127.0.0.1:5000/predict", { review });
      setResult(response.data.prediction);
    } catch (error) {
      setResult("Error processing request.");
      console.error(error);
    }
  };

  return (
    <div className="container">
      <h2>Fake Review Detection</h2>
      <textarea
        value={review}
        onChange={(e) => setReview(e.target.value)}
        placeholder="Enter your review here..."
      />
      <button onClick={checkReview}>Check Review</button>
      <p>{result}</p>
    </div>
  );
}

export default App;
