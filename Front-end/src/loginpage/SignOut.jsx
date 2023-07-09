import React from 'react';
import {useNavigate} from 'react-router-dom'

export default function SignOut() {
  const navigate = useNavigate()
  return(
    <>
    <div>Home Page</div>
    <button onClick={() => navigate(-1)}>Signin</button>
    </>
  );
}
