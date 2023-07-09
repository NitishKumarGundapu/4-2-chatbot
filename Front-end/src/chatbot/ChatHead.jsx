import React from "react";
import "./main.css";
import Capture from './images/Capture.PNG';
/*
function ChatHead(){
    return (
        <div className="chat-head">
            <label className="title-chat-1" title="Sara">Lenskart Chatbot</label>
        </div>
    )
}
*/
/*
function ChatHead(){
    return(
        <div className="chat-head">
            <div className="right-brand-icon">
                <span className="right-icon">L</span>
            </div>
            <div className="fbots-chat-header-title">
                <label className="title-chat-1" title="Lenskart Chatbot">Lenskart Chatbot</label>
            </div>
        
        </div>
        )
}
*/
/*
function ChatHead()
{
    return(
        <div className="e17_15">
            <div className="e17_12"></div><span className="e17_13">I am here to help you</span><span
                className="e17_11">Lenskart
                ChatBot</span>
        </div>
    )
}

*/
function ChatHead() {
    // Import result is the URL of your image
    return (
        <div className="chat-header">
            <img src={Capture} width={300} height={75} alt="Logo" />
        </div>  );
  }
  

export default ChatHead
