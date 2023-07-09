import React from "react";
import {useState,useEffect} from "react";
import api from "../api/retriver"
import a from "../api/language"
import "./main.css";

const text11 = [
  "1.Place an order online",
  "2.Place an order offline"
]

const Options1 = (props) => {
  const [text1,setText] = useState({})  
  const get_translation = async (message,language) => {
    const response = await api.get("/translation",{
      params : {
      msg: ''.concat(message),
      lang: language
    }})
    return setText(response.data);
  };

   useEffect(() => {get_translation(text11,a.getLanguage())},[]);
  const options= [
    {
      text: text1[0],
      handler: props.actionProvider.place_an_order_online,
      id: 1,
    },
    { 
      text: text1[1], 
      handler: props.actionProvider.place_an_order_offline, 
      id: 2,
    },
  ];

  const buttonsMarkup = options.map((option) => (
    <button key={option.id} onClick={option.handler} className="option-button">
      {option.text}
    </button>
  ));

  return <div className="order-options-container">{buttonsMarkup}</div>;
};

export default Options1;
