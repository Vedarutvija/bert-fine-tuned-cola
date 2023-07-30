

const firebaseConfig = {
    apiKey: "AIzaSyDpjbP-Vi6FUEIb0mAoj5dPiLKx4xZwjGw",
    authDomain: "asana-eddc2.firebaseapp.com",
    databaseURL: "https://asana-eddc2-default-rtdb.firebaseio.com",
    projectId: "asana-eddc2",
    storageBucket: "asana-eddc2.appspot.com",
    messagingSenderId: "331037292315",
    appId: "1:331037292315:web:58a5edf997c3b73d03c5be"
  };
firebase.initializeApp(firebaseConfig);


var contactFormDB = firebase.database().ref("contactForm");

// listen for submit
document.getElementById('contactForm').addEventListener("submit",submitForm);

function submitForm(e){
    e.preventDefault();
    var name = getElementVal('name');
    var email = getElementVal('email');
    var subject = getElementVal('subject');
    var message = getElementVal('message');

    saveMessages(name, email, subject, message);
        document.querySelector(".alert").style.display ="block";
    setTimeout(()=>{
        document.querySelector(".alert").style.display = "none";
    }, 3000);

    document.getElementById("contactForm").rest();
}
const saveMessages = (name, email, subject, message)=> {
    var newContactForm = contactFormDB.push();
    newContactForm.set({
        name: name,
        email: email,
        subject: subject,
        message: message,
    });
};
const getElementVal = (id) =>{
    return document.getElementById(id).value;
};


