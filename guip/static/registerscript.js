const passwordInput = document.getElementById('password');
const confirmpassInput = document.getElementById('confirm-password');
const showPasswordCheckbox = document.getElementById('show-password');

showPasswordCheckbox.addEventListener('change', function() {
    if (this.checked) {
        passwordInput.type = 'text';
        confirmpassInput.type ='text';
    } else {
        passwordInput.type = 'password';
        confirmpassInput.type ='password';
    }
});
