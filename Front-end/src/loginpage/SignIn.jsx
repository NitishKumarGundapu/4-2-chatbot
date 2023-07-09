/*
import React from 'react';
import {useNavigate} from 'react-router-dom'

export default function SignIn() {
  const navigate = useNavigate()
  return(
    <>
    <div>Home Page</div>
    <button onClick={() => navigate('chatbot')}>Signin</button>
    </>
  );
}
*/

//import ReactDOM from "react-dom";
import React, { useState } from "react";
import {useNavigate} from 'react-router-dom'

import "./styles.css";

function SignIn() {

  const navigate = useNavigate()
  // React States
  const [errorMessages, setErrorMessages] = useState({});
  const [isSubmitted, setIsSubmitted] = useState(false);

  // User Login info
  const database = [
    {
      username: "user1@lenskart.in",
      password: "pass1"
    },
    {
      username: "user2@lenskart.in",
      password: "pass2"
    },
    {
      username: "user3@lenskart.in",
      password: "pass3"
    }
  ];

  const errors = {
    uname: "invalid username",
    pass: "invalid password"
  };

  const handleSubmit = (event) => {
    event.preventDefault();
    var { uname, pass } = document.forms[0];
    const userData = database.find((user) => user.username === uname.value);

    if (userData) {
      if (userData.password !== pass.value) {
        setErrorMessages({ name: "pass", message: errors.pass });
      } else {
        setIsSubmitted(true);
      }
    } else {
      setErrorMessages({ name: "uname", message: errors.uname });
    }
  };

  // Generate JSX code for error message
  const renderErrorMessage = (name) =>
    name === errorMessages.name && (
      <div className="error">{errorMessages.message}</div>
    );

    // JSX code for login form
  const renderForm = (
    <div className="form">
      <form onSubmit={handleSubmit}>
        <div className="input-container">
          <label>Username </label>
          <input type="text" name="uname" required />
          {renderErrorMessage("uname")}
        </div>
        <br></br>
        <div className="input-container">
          <label>Password </label>
          <input type="password" name="pass" required />
          {renderErrorMessage("pass")}
        </div>
        <br></br>
        <div className="button-container">
          <input type="submit" />
        </div>
      </form>
    </div>
  );

  return (
    <div className="app">
      <div className="login-form">
        <div className="title">Login to Your Account</div><br></br>
        {isSubmitted ? navigate('chatbot') : renderForm}
      </div>
    </div>
  );
}

export default SignIn;
