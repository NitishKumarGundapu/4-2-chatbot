import React from "react";
import {useState,useEffect} from "react";
import api from "../api/retriver"
import a from "../api/language"
import "./main.css";

const text1 = [
  "1. Steps to place an order online or offline stores",
  "2. Finding nearest Lenskart Store in your surroundings",
  "3. Cancellation of Ordered Items with order number",
  "4. Making an appointment for a home eye test and try-on",
  "5. Select different input and display language",
  "6. Logout from your lenskart account"
]

const Options = (props) => {
  const [text,setText] = useState([]) 
  const get_translation = async (message,language) => {
    const response = await api.get("/translation",{
      params : {
      msg: ''.concat(message),
      lang: language
    }})
    return setText(response.data);
  };

  useEffect(() => {get_translation(text1,a.getLanguage())},[]);

  var text_0 = text[0]
  var text_1 = text[1]
  var text_2 = text[2]
  var text_3 = text[3]
  var text_4 = text[4]
  var text_5 = text[5]

  const options = [
    {
      text: text_0,
      handler: props.actionProvider.starting_option_1,
      id: 1,
    },
    { 
      text: text_1, 
      handler: props.actionProvider.starting_option_2, 
      id: 2,
    },
    { 
      text: text_2, 
      handler: props.actionProvider.starting_option_3, 
      id: 3,
    },
    { 
      text: text_3, 
      handler: props.actionProvider.starting_option_4, 
      id: 4,
    },
    { 
      text: text_4, 
      handler: props.actionProvider.starting_option_5, 
      id: 5,
    },
    { 
      text: text_5, 
      handler: props.actionProvider.starting_option_6, 
      id: 6,
    }
  ];

  const buttonsMarkup = options.map((option) => (
    <button key={option.id} onClick={option.handler} className="option-button">
      {option.text}
    </button>
  ));

  return <div className="options-container">{buttonsMarkup}</div>;
};

export default Options;
