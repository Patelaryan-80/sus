// Main JavaScript for SURVION 

document.addEventListener('DOMContentLoaded', function() {
    // Create animated particles for hero background
    const heroSection = document.querySelector('.hero');
    const heroParticles = document.querySelector('.hero-particles');
    
    if (heroSection && heroParticles) {
        // Create floating particles
        for (let i = 0; i < 20; i++) {
            createParticle(heroParticles);
        }
    }
    
    function createParticle(container) {
        const particle = document.createElement('div');
        particle.classList.add('particle');
        
        // Randomize position, size and animation
        const size = Math.random() * 5 + 2; // 2-7px
        const posX = Math.random() * 100; // 0-100%
        const posY = Math.random() * 100; // 0-100%
        const duration = Math.random() * 20 + 10; // 10-30s
        const delay = Math.random() * 5; // 0-5s
        const opacity = Math.random() * 0.5 + 0.1; // 0.1-0.6
        
        particle.style.cssText = `
            position: absolute;
            width: ${size}px;
            height: ${size}px;
            left: ${posX}%;
            top: ${posY}%;
            background: ${Math.random() > 0.5 ? 'var(--primary-color)' : 'var(--accent-color)'};
            border-radius: 50%;
            opacity: ${opacity};
            filter: blur(1px);
            animation: float ${duration}s infinite alternate ease-in-out ${delay}s;
            z-index: 1;
        `;
        
        container.appendChild(particle);
    }
    
    // Add floating animation to CSS if not exists
    if (!document.querySelector('#particle-styles')) {
        const styleSheet = document.createElement('style');
        styleSheet.id = 'particle-styles';
        styleSheet.textContent = `
            @keyframes float {
                0% {
                    transform: translateY(0) translateX(0);
                    opacity: 0.1;
                }
                50% {
                    opacity: 0.5;
                }
                100% {
                    transform: translateY(-100px) translateX(20px);
                    opacity: 0.2;
                }
            }
        `;
        document.head.appendChild(styleSheet);
    }

    // Enhanced typing effect for hero text
    const typingElement = document.querySelector('.typing-effect');
    if(typingElement) {
        const text = typingElement.textContent;
        typingElement.setAttribute('data-text', text); // Store original text
        typingElement.textContent = '';
        let i = 0;
        
        function typeEffect() {
            if (i < text.length) {
                typingElement.textContent += text.charAt(i);
                i++;
                setTimeout(typeEffect, Math.random() * 80 + 50); // Randomized typing speed
            } else {
                setTimeout(() => {
                    // Add blinking cursor effect
                    typingElement.classList.add('cursor-blink');
                }, 1000);
            }
        }
        
        setTimeout(typeEffect, 800);
    }
    
    // Add cursor blinking style
    if (!document.querySelector('#cursor-blink-style')) {
        const cursorStyle = document.createElement('style');
        cursorStyle.id = 'cursor-blink-style';
        cursorStyle.textContent = `
            .cursor-blink::after {
                content: '|';
                color: var(--primary-color);
                animation: cursor-blink 1s infinite;
                margin-left: 5px;
            }
            
            @keyframes cursor-blink {
                0%, 100% { opacity: 1; }
                50% { opacity: 0; }
            }
        `;
        document.head.appendChild(cursorStyle);
    }
    
    // Glitch effect
    const glitchOverlay = document.querySelector('.glitch-overlay');
    const links = document.querySelectorAll('a');
    const buttons = document.querySelectorAll('.btn');
    
    // Apply glitch effect on interaction
    function applyGlitch() {
        glitchOverlay.classList.add('glitch-active');
        setTimeout(() => {
            glitchOverlay.classList.remove('glitch-active');
        }, 300);
    }
    
    // Apply glitch effect randomly
    function randomGlitch() {
        if (Math.random() > 0.997) {
            applyGlitch();
        }
    }
    
    // Set interval for random glitch effect
    setInterval(randomGlitch, 100);
    
    // Add glitch on button hover
    buttons.forEach(btn => {
        btn.addEventListener('mouseenter', applyGlitch);
    });
    
    // Mobile navigation toggle
    const mobileToggle = document.querySelector('.mobile-toggle');
    const navLinks = document.querySelector('.nav-links');
    const navOverlay = document.querySelector('.nav-overlay');
    
    if(mobileToggle) {
        mobileToggle.addEventListener('click', function() {
            mobileToggle.classList.toggle('active');
            navLinks.classList.toggle('show');
            navOverlay.classList.toggle('active');
            applyGlitch();
        });
    }
    
    if(navOverlay) {
        navOverlay.addEventListener('click', function() {
            mobileToggle.classList.remove('active');
            navLinks.classList.remove('show');
            navOverlay.classList.remove('active');
        });
    }
    
    // Stats Animation - Fixed to work properly
    const statItems = document.querySelectorAll('.stat-item h3');
    
    function animateValue(element, start, end, duration) {
        let startTimestamp = null;
        const step = (timestamp) => {
            if (!startTimestamp) startTimestamp = timestamp;
            const progress = Math.min((timestamp - startTimestamp) / duration, 1);
            
            // Handle different formats of values
            if (element.innerText.includes('%')) {
                element.innerText = parseFloat((progress * (end - start) + start).toFixed(1)) + '%';
            } else if (element.innerText.includes('s')) {
                element.innerText = parseFloat((progress * (end - start) + start).toFixed(2)) + 's';
            } else if (element.innerText.includes('/')) {
                element.innerText = '24/7';
            } else if (element.innerText.includes('+')) {
                element.innerText = Math.floor(progress * (end - start) + start) + '+';
            } else {
                element.innerText = Math.floor(progress * (end - start) + start);
            }
            
            if (progress < 1) {
                window.requestAnimationFrame(step);
            }
        };
        window.requestAnimationFrame(step);
    }
    
    function isInViewport(element) {
        const rect = element.getBoundingClientRect();
        return (
            rect.top >= 0 &&
            rect.left >= 0 &&
            rect.bottom <= (window.innerHeight || document.documentElement.clientHeight) &&
            rect.right <= (window.innerWidth || document.documentElement.clientWidth)
        );
    }
    
    function checkStatsVisibility() {
        statItems.forEach(stat => {
            if (isInViewport(stat) && !stat.classList.contains('animated')) {
                stat.classList.add('animated');
                
                let value = stat.getAttribute('data-value');
                let startValue = 0;
                let endValue = parseFloat(value);
                
                animateValue(stat, startValue, endValue, 2000);
            }
        });
    }
    
    // Initialize animation on scroll
    window.addEventListener('scroll', checkStatsVisibility);
    // Also check on page load
    checkStatsVisibility();
    
    // Custom cursor implementation
    const cursor = document.createElement('div');
    cursor.className = 'cursor-dot';
    document.body.appendChild(cursor);
    
    const cursorOutline = document.createElement('div');
    cursorOutline.className = 'cursor-outline';
    document.body.appendChild(cursorOutline);
    
    document.addEventListener('mousemove', e => {
        cursor.style.left = e.clientX + 'px';
        cursor.style.top = e.clientY + 'px';
        
        cursorOutline.style.left = e.clientX + 'px';
        cursorOutline.style.top = e.clientY + 'px';
    });
    
    document.addEventListener('mousedown', () => {
        cursor.style.transform = 'scale(0.7)';
        cursorOutline.style.transform = 'scale(0.7)';
    });
    
    document.addEventListener('mouseup', () => {
        cursor.style.transform = 'scale(1)';
        cursorOutline.style.transform = 'scale(1)';
    });
    
    // Grow cursor on interactive elements
    const interactiveElements = document.querySelectorAll('a, button, .btn, input, select, textarea');
    interactiveElements.forEach(el => {
        el.addEventListener('mouseenter', () => {
            cursor.style.transform = 'scale(1.5)';
            cursorOutline.style.transform = 'scale(1.5)';
            cursorOutline.style.borderColor = 'var(--primary-color)';
        });
        
        el.addEventListener('mouseleave', () => {
            cursor.style.transform = 'scale(1)';
            cursorOutline.style.transform = 'scale(1)';
            cursorOutline.style.borderColor = 'var(--primary-color)';
        });
    });
    
    // Testimonial slider
    let currentSlide = 0;
    const testimonials = document.querySelectorAll('.testimonial');
    
    if(testimonials.length > 0) {
        function showSlide(index) {
            testimonials.forEach(slide => {
                slide.style.display = 'none';
                slide.classList.remove('active');
            });
            
            if(index >= testimonials.length) {
                currentSlide = 0;
            } else if(index < 0) {
                currentSlide = testimonials.length - 1;
            } else {
                currentSlide = index;
            }
            
            testimonials[currentSlide].style.display = 'block';
            
            // Add animation class after display is set
            setTimeout(() => {
                testimonials[currentSlide].classList.add('active');
            }, 10);
        }
        
        // Initial display
        showSlide(currentSlide);
        
        // Auto-advance testimonials
        setInterval(() => {
            showSlide(currentSlide + 1);
        }, 7000);
    }
    
    // Animate elements when they come into view
    const animateOnScroll = function() {
        const elements = document.querySelectorAll('.animate-on-scroll');
        
        elements.forEach(element => {
            const elementPosition = element.getBoundingClientRect().top;
            const screenPosition = window.innerHeight / 1.3;
            
            if(elementPosition < screenPosition) {
                element.classList.add('animate');
            }
        });
    };
    
    // Run on scroll
    window.addEventListener('scroll', animateOnScroll);
    
    // Run once on page load
    animateOnScroll();
    
    // Add animation classes to various elements
    document.querySelectorAll('.feature-card, .tech-card, .step, .dashboard-card').forEach(
        (el, index) => {
            el.classList.add('animate-on-scroll');
            el.style.transitionDelay = `${index * 0.1}s`;
        }
    );
});
