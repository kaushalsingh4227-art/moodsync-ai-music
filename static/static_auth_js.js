function showSignup() {
    document.getElementById("loginForm").style.display = "none";
    document.getElementById("signupForm").style.display = "block";
}

function showLogin() {
    document.getElementById("loginForm").style.display = "block";
    document.getElementById("signupForm").style.display = "none";
}

function togglePass(id) {
    let field = document.getElementById(id);
    field.type = field.type === "password" ? "text" : "password";
}