"""
用户认证模块
"""

import hashlib
import streamlit as st
from typing import Optional, Dict
from database.models import db_manager

class AuthManager:
    """认证管理器"""
    
    def __init__(self):
        self.db = db_manager
    
    def hash_password(self, password: str) -> str:
        """
        对密码进行哈希处理
        
        Args:
            password: 原始密码
            
        Returns:
            str: 哈希后的密码
        """
        return hashlib.sha256(password.encode()).hexdigest()
    
    def register_user(self, username: str, password: str, email: str = None) -> Dict:
        """
        注册新用户
        
        Args:
            username: 用户名
            password: 密码
            email: 邮箱
            
        Returns:
            Dict: 注册结果
        """
        # 验证输入
        if not username or not password:
            return {
                'success': False,
                'message': '用户名和密码不能为空'
            }
        
        if len(password) < 6:
            return {
                'success': False,
                'message': '密码长度至少6位'
            }
        
        # 哈希密码
        password_hash = self.hash_password(password)
        
        # 添加到数据库
        success = self.db.add_user(username, password_hash, email)
        
        if success:
            # 记录系统日志
            import json as _json
            self.db.add_system_log(
                user_id=None,
                action="user_registration",
                details=_json.dumps({"msg": "新用户注册", "username": username}, ensure_ascii=False),
                status="success"
            )
            
            return {
                'success': True,
                'message': '注册成功！'
            }
        else:
            return {
                'success': False,
                'message': '用户名已存在，请选择其他用户名'
            }
    
    def login_user(self, username: str, password: str) -> Dict:
        """
        用户登录
        
        Args:
            username: 用户名
            password: 密码
            
        Returns:
            Dict: 登录结果
        """
        # 验证输入
        if not username or not password:
            return {
                'success': False,
                'message': '用户名和密码不能为空'
            }
        
        # 哈希密码
        password_hash = self.hash_password(password)
        
        # 验证用户
        user_id = self.db.verify_user(username, password_hash)
        
        if user_id:
            # 获取用户信息
            user_info = self.db.get_user_by_id(user_id)
            
            # 记录系统日志
            import json as _json
            self.db.add_system_log(
                user_id=user_id,
                action="user_login",
                details=_json.dumps({"msg": "用户登录", "username": username}, ensure_ascii=False),
                status="success"
            )
            
            return {
                'success': True,
                'message': '登录成功！',
                'user_id': user_id,
                'user_info': user_info
            }
        else:
            return {
                'success': False,
                'message': '用户名或密码错误'
            }
    
    def is_logged_in(self) -> bool:
        """
        检查用户是否已登录
        
        Returns:
            bool: 是否已登录
        """
        return 'user_id' in st.session_state
    
    def get_current_user(self) -> Optional[Dict]:
        """
        获取当前登录用户信息
        
        Returns:
            Optional[Dict]: 用户信息
        """
        if self.is_logged_in():
            user_id = st.session_state['user_id']
            return self.db.get_user_by_id(user_id)
        return None
    
    def logout_user(self):
        """
        用户登出
        """
        if self.is_logged_in():
            user_id = st.session_state['user_id']
            username = st.session_state.get('username', 'Unknown')
            
            # 记录系统日志
            import json as _json
            self.db.add_system_log(
                user_id=user_id,
                action="user_logout",
                details=_json.dumps({"msg": "用户登出", "username": username}, ensure_ascii=False),
                status="success"
            )
        
        # 清除会话状态
        for key in ['user_id', 'username', 'user_info']:
            if key in st.session_state:
                del st.session_state[key]

# 创建全局认证管理器实例
auth_manager = AuthManager()

def login_page():
    """登录页面"""
    st.title("🎭 情绪音乐助手 - 登录")
    # 若已登录则跳转主页面
    if auth_manager.is_logged_in():
        st.session_state['page'] = 'main'
        st.rerun()
    
    # 创建登录表单
    with st.form("login_form"):
        username = st.text_input("用户名")
        password = st.text_input("密码", type="password")
        submit_button = st.form_submit_button("登录")
        
        if submit_button:
            result = auth_manager.login_user(username, password)
            
            if result['success']:
                # 保存用户信息到会话状态
                st.session_state['user_id'] = result['user_id']
                st.session_state['username'] = result['user_info']['username']
                st.session_state['user_info'] = result['user_info']
                st.session_state['is_admin'] = (result['user_info']['username'] == 'asd')
                st.session_state['page'] = 'main'
                
                st.success(result['message'])
                st.rerun()
            else:
                st.error(result['message'])
    
    # 注册链接
    st.markdown("---")
    st.markdown("还没有账号？")
    if st.button("注册新账号"):
        st.session_state['page'] = 'register'
        st.rerun()

def register_page():
    """注册页面"""
    st.title("🎭 情绪音乐助手 - 注册")
    
    # 创建注册表单
    with st.form("register_form"):
        username = st.text_input("用户名")
        password = st.text_input("密码", type="password")
        confirm_password = st.text_input("确认密码", type="password")
        email = st.text_input("邮箱（可选）")
        submit_button = st.form_submit_button("注册")
        
        if submit_button:
            # 验证密码
            if password != confirm_password:
                st.error("两次输入的密码不一致")
            else:
                result = auth_manager.register_user(username, password, email)
                
                if result['success']:
                    st.success(result['message'])
                    st.info("请返回登录页面进行登录")
                else:
                    st.error(result['message'])
    
    # 返回登录链接
    st.markdown("---")
    st.markdown("已有账号？")
    if st.button("返回登录"):
        st.session_state['page'] = 'login'
        st.rerun()

def show_user_info():
    """显示用户信息"""
    if auth_manager.is_logged_in():
        user_info = auth_manager.get_current_user()
        if user_info:
            st.sidebar.markdown("---")
            st.sidebar.markdown(f"👤 **{user_info['username']}**")
            st.sidebar.markdown(f"📧 {user_info['email'] or '未设置邮箱'}")
            if st.session_state.get('is_admin'):
                st.sidebar.markdown("🛡️ 管理员")
            
            if st.sidebar.button("登出"):
                auth_manager.logout_user()
                st.session_state['page'] = 'login'
                st.rerun()

def require_login():
    """
    要求用户登录的装饰器
    如果用户未登录，自动跳转到登录页面
    """
    if not auth_manager.is_logged_in():
        st.session_state['page'] = 'login'
        st.rerun() 