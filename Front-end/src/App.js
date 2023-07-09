import React from "react";
import "./App.css";
import SignIn  from './loginpage/SignIn'
import ChatBot from './chatbot/ChatBot'
//import {useNavigate} from 'react-router-dom'


import {Routes,Route} from "react-router-dom";

function App() {
  return (
    <div className="wrapper">
      <Routes>
        <Route path="/" element={<SignIn />} />
        <Route path="/chatbot" element={<ChatBot />} />
      </Routes>
    </div>
  );
}

export default App;