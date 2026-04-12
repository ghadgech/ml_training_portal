import React, { useState } from 'react';
import { Plus, Edit2, Trash2, AlertCircle } from 'lucide-react';
import { Product } from '../types';
import { getProducts, saveProduct, deleteProduct } from '../storage';
import '../styles/InventoryManager.css';

export default function InventoryManager() {
  const [products, setProducts] = useState<Product[]>(getProducts());
  const [showForm, setShowForm] = useState(false);
  const [editingId, setEditingId] = useState<string | null>(null);
  const [formData, setFormData] = useState<Partial<Product>>({
    name: '',
    sku: '',
    category: '',
    cost: 0,
    sellingPrice: 0,
    stock: 0,
  });

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) => {
    const { name, value } = e.target;
    setFormData({
      ...formData,
      [name]: name.includes('Price') || name === 'cost' || name === 'stock' ? parseFloat(value) || 0 : value,
    });
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();

    if (!formData.name || !formData.sku || !formData.category) {
      alert('Please fill all required fields');
      return;
    }

    if (editingId) {
      const product: Product = {
        id: editingId,
        name: formData.name,
        sku: formData.sku,
        category: formData.category,
        cost: formData.cost || 0,
        sellingPrice: formData.sellingPrice || 0,
        stock: formData.stock || 0,
        expiryDate: formData.expiryDate,
      };
      saveProduct(product);
    } else {
      const newProduct: Product = {
        id: `PROD-${Date.now()}`,
        name: formData.name,
        sku: formData.sku,
        category: formData.category,
        cost: formData.cost || 0,
        sellingPrice: formData.sellingPrice || 0,
        stock: formData.stock || 0,
        expiryDate: formData.expiryDate,
      };
      saveProduct(newProduct);
    }

    setProducts(getProducts());
    resetForm();
  };

  const resetForm = () => {
    setFormData({
      name: '',
      sku: '',
      category: '',
      cost: 0,
      sellingPrice: 0,
      stock: 0,
    });
    setEditingId(null);
    setShowForm(false);
  };

  const handleEdit = (product: Product) => {
    setFormData(product);
    setEditingId(product.id);
    setShowForm(true);
  };

  const handleDelete = (id: string) => {
    if (confirm('Are you sure you want to delete this product?')) {
      deleteProduct(id);
      setProducts(getProducts());
    }
  };

  const lowStockItems = products.filter(p => p.stock < 10);
  const totalInventoryValue = products.reduce((sum, p) => sum + p.cost * p.stock, 0);

  return (
    <div className="inventory-manager">
      <div className="inventory-header">
        <h2 className="card-title">Inventory Management</h2>
        <button className="btn-primary button" onClick={() => setShowForm(true)}>
          <Plus size={18} /> Add Product
        </button>
      </div>

      {/* Stats */}
      <div className="grid-3">
        <div className="stat-card">
          <div className="stat-label">Total Products</div>
          <div className="stat-value">{products.length}</div>
        </div>
        <div className="stat-card">
          <div className="stat-label">Inventory Value</div>
          <div className="stat-value">₹{totalInventoryValue.toLocaleString()}</div>
        </div>
        <div className="stat-card">
          <div className="stat-label">Low Stock Alert</div>
          <div className="stat-value">{lowStockItems.length}</div>
        </div>
      </div>

      {/* Low Stock Alert */}
      {lowStockItems.length > 0 && (
        <div className="alert alert-warning">
          <AlertCircle size={20} />
          <div>
            <strong>Low Stock Alert:</strong> {lowStockItems.map(p => p.name).join(', ')}
          </div>
        </div>
      )}

      {/* Form Modal */}
      {showForm && (
        <div className="modal-overlay" onClick={() => resetForm()}>
          <div className="modal" onClick={e => e.stopPropagation()}>
            <h3 className="modal-title">
              {editingId ? 'Edit Product' : 'Add New Product'}
            </h3>

            <form onSubmit={handleSubmit}>
              <div className="grid-2">
                <div className="form-group">
                  <label className="form-label">Product Name *</label>
                  <input
                    type="text"
                    name="name"
                    className="form-input"
                    value={formData.name || ''}
                    onChange={handleInputChange}
                    required
                  />
                </div>
                <div className="form-group">
                  <label className="form-label">SKU *</label>
                  <input
                    type="text"
                    name="sku"
                    className="form-input"
                    value={formData.sku || ''}
                    onChange={handleInputChange}
                    required
                  />
                </div>
                <div className="form-group">
                  <label className="form-label">Category *</label>
                  <select
                    name="category"
                    className="form-select"
                    value={formData.category || ''}
                    onChange={handleInputChange}
                    required
                  >
                    <option value="">Select Category</option>
                    <option value="Supplements">Supplements</option>
                    <option value="Oils">Oils</option>
                    <option value="Powder">Powder</option>
                    <option value="Tablets">Tablets</option>
                    <option value="Creams">Creams</option>
                    <option value="Other">Other</option>
                  </select>
                </div>
                <div className="form-group">
                  <label className="form-label">Cost Price (₹)</label>
                  <input
                    type="number"
                    name="cost"
                    className="form-input"
                    value={formData.cost || 0}
                    onChange={handleInputChange}
                  />
                </div>
                <div className="form-group">
                  <label className="form-label">Selling Price (₹)</label>
                  <input
                    type="number"
                    name="sellingPrice"
                    className="form-input"
                    value={formData.sellingPrice || 0}
                    onChange={handleInputChange}
                  />
                </div>
                <div className="form-group">
                  <label className="form-label">Stock Quantity</label>
                  <input
                    type="number"
                    name="stock"
                    className="form-input"
                    value={formData.stock || 0}
                    onChange={handleInputChange}
                  />
                </div>
                <div className="form-group">
                  <label className="form-label">Expiry Date</label>
                  <input
                    type="date"
                    name="expiryDate"
                    className="form-input"
                    value={formData.expiryDate || ''}
                    onChange={handleInputChange}
                  />
                </div>
              </div>

              {formData.cost && formData.sellingPrice && (
                <p className="margin-info">
                  Margin: ₹{(formData.sellingPrice - formData.cost).toFixed(2)} per unit ({(((formData.sellingPrice - formData.cost) / formData.cost) * 100).toFixed(1)}%)
                </p>
              )}

              <div className="modal-footer">
                <button type="button" className="btn-secondary button" onClick={resetForm}>
                  Cancel
                </button>
                <button type="submit" className="btn-primary button">
                  {editingId ? 'Update' : 'Add'} Product
                </button>
              </div>
            </form>
          </div>
        </div>
      )}

      {/* Products Table */}
      <div className="card">
        <table className="table">
          <thead>
            <tr>
              <th>Name</th>
              <th>SKU</th>
              <th>Category</th>
              <th>Cost</th>
              <th>Selling Price</th>
              <th>Margin %</th>
              <th>Stock</th>
              <th>Expiry</th>
              <th>Actions</th>
            </tr>
          </thead>
          <tbody>
            {products.map(product => {
              const margin = ((product.sellingPrice - product.cost) / product.cost) * 100;
              return (
                <tr key={product.id} className={product.stock < 10 ? 'low-stock' : ''}>
                  <td>{product.name}</td>
                  <td>{product.sku}</td>
                  <td>{product.category}</td>
                  <td>₹{product.cost}</td>
                  <td>₹{product.sellingPrice}</td>
                  <td>{margin.toFixed(1)}%</td>
                  <td>
                    <span className={product.stock < 10 ? 'stock-warning' : ''}>
                      {product.stock}
                    </span>
                  </td>
                  <td>{product.expiryDate || '-'}</td>
                  <td>
                    <button
                      className="btn-secondary button"
                      onClick={() => handleEdit(product)}
                      title="Edit"
                    >
                      <Edit2 size={16} />
                    </button>
                    <button
                      className="btn-danger button"
                      onClick={() => handleDelete(product.id)}
                      title="Delete"
                    >
                      <Trash2 size={16} />
                    </button>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}
